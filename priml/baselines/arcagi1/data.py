"""ARC-AGI tasks, served from device memory.

The prepared dataset is a three-level hierarchy, which is what makes ARC
different from a flat dataset::

    group  -- one ARC task (a rule)
      puzzle -- one held-out input for that task
        example -- one augmented view of that puzzle

On disk::

    all__inputs.npy             [n_examples, 900] input tokens
    all__labels.npy             [n_examples, 900] target tokens
    all__puzzle_indices.npy     [n_puzzles + 1]   example offsets per puzzle
    all__group_indices.npy      [n_groups + 1]    puzzle offsets per task
    all__puzzle_identifiers.npy [n_puzzles]       per-puzzle task id
    dataset.json                shape and vocabulary metadata

Tokens are ``0`` pad, ``1`` a blank marker, and ``2``-``11`` the ten ARC
colors. Grids are padded to 30x30 because ARC grids vary in size and the model
needs one shape.

Training samples by TASK, not by row: each batch draws a random task, then a
random puzzle from it, then random augmented views of that puzzle. Sampling
rows uniformly instead would over-weight tasks that happen to have more
puzzles, and the benchmark weights every task equally.

``scripts/prepare_data.py`` builds the arrays; this module only reads them, so
constructing a config never touches the network.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self, override

import json
import logging
import math

from configgle import Fig
from torch import Tensor

import numpy as np
import torch

from priml.paths import resolve_working_dir
from priml.runtime import get_device


logger = logging.getLogger(__name__)


class ArcData:
    """ARC tasks held in device memory, yielding ``media`` / ``label`` batches.

    Every batch is exactly ``batch_size`` rows: a short final batch is padded
    with zero rows and reports how many are real, so downstream tensor shapes
    never change mid-epoch. Batches also carry ``puzzle_identifiers``, which
    the per-task prefix and the pass@K metric both read.

    Raises:
      FileNotFoundError: If the prepared arrays are absent. Run
        ``uv --quiet run --frozen python -m
        priml.baselines.arcagi1.scripts.prepare_data`` first.

    """

    class Config(Fig["ArcData"]):
        """Where the prepared arrays live, and how batches are drawn."""

        base_dir: Path | str | None = None
        """Resource root supplied during parent finalization."""

        working_dir: Path | str = "/datasets/arcagi1"
        """Directory holding the ``train/`` and ``test/`` splits.

        Resolved beneath ``base_dir`` at finalize, so it names a location
        within the resource root rather than an absolute filesystem path."""

        batch_size: int = 256
        """Examples per training batch."""

        eval_batch_size: int | None = None
        """Examples per evaluation batch; ``None`` reuses ``batch_size``."""

        device: str = "auto"
        """Device holding the resident arrays ("auto" picks the best)."""

        seed: int = 0
        """Seeds the task-sampling stream.

        Fixed rather than optional because sampling is hierarchical: a run has
        to be able to replay which tasks and which augmented views it saw."""

        num_tasks: int | None = None
        """Training tasks to load; ``None`` loads all of them."""

        num_eval_tasks: int | None = None
        """Evaluation tasks to load; ``None`` loads all of them.

        The full evaluation split is large and every task contributes many
        augmented views, so mid-training evaluation normally reads a prefix of
        it and the reported number comes from an uncapped final pass. The two
        populations are not comparable."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.dataset_dir = Path(config.working_dir)
        self.batch_size = config.batch_size
        self.eval_batch_size = config.eval_batch_size or config.batch_size
        # Completed passes, persisted across resume so a restored run continues
        # the sampling sequence instead of replaying the first pass.
        self._passes = 0
        self._live: _ArcBatches | None = None

    def train_dataloader(self) -> _ArcBatches:
        """Build the re-iterable training stream."""
        # Snapshot any prior stream's counter first, so re-creating the loader
        # continues the sequence rather than restarting it.
        if self._live is not None:
            self._passes = self._live.passes
        stream = _ArcBatches(
            dataset_dir=self.dataset_dir,
            device=self.config.device,
            batch_size=self.batch_size,
            split="train",
            sample_by_task=True,
            num_tasks=self.config.num_tasks,
            seed=self.config.seed,
            passes=self._passes,
        )
        self._live = stream
        return stream

    def eval_dataloader(self) -> _ArcBatches:
        """Build the evaluation stream: every view of every task, in order.

        Evaluation must see every augmented view, because pass@K votes across
        them -- sampling here would discard the ballots.
        """
        return _ArcBatches(
            dataset_dir=self.dataset_dir,
            device=self.config.device,
            batch_size=self.eval_batch_size,
            split="test",
            sample_by_task=False,
            num_tasks=self.config.num_eval_tasks,
            seed=self.config.seed,
            passes=0,
        )

    def state_dict(self) -> dict[str, Any]:
        """Snapshot the completed-pass count."""
        return {"passes": self._live.passes if self._live is not None else self._passes}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""
        if "passes" in state_dict:
            self._passes = int(state_dict["passes"])
            if self._live is not None:
                self._live.passes = self._passes


class _ArcBatches:
    """One split, resident on device, iterated in fixed-size batches."""

    def __init__(
        self,
        *,
        dataset_dir: Path,
        device: torch.device | str,
        batch_size: int,
        split: str,
        sample_by_task: bool,
        num_tasks: int | None,
        seed: int,
        passes: int,
    ) -> None:
        data = _load_split(dataset_dir, split)
        groups = data["group_indices"]
        puzzles = data["puzzle_indices"]
        if num_tasks is not None and num_tasks < len(groups) - 1:
            groups = groups[: num_tasks + 1]
            puzzles = puzzles[: int(groups[-1]) + 1]
            rows = int(puzzles[-1])
        else:
            rows = len(data["inputs"])

        self.device = get_device(device)
        self.inputs: Tensor = data["inputs"][:rows].to(self.device)
        self.labels: Tensor = data["labels"][:rows].to(self.device)
        self.groups: np.ndarray = groups
        self.puzzles: np.ndarray = puzzles
        self.identifiers: np.ndarray = data["puzzle_identifiers"][: len(puzzles) - 1]
        self.ignore_label_id = int(data["ignore_label_id"])
        self.batch_size = batch_size
        self.sample_by_task = sample_by_task
        self.seed = seed
        self.passes = passes

    @property
    def num_tasks(self) -> int:
        """Tasks in this split."""
        return len(self.groups) - 1

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield batches: sampled by task for training, in order for eval."""
        if self.sample_by_task:
            yield from self._iter_sampled()
        else:
            yield from self._iter_ordered()

    def __len__(self) -> int:
        """Batches per pass, counting a short final batch."""
        return math.ceil(int(self.puzzles[-1]) / self.batch_size)

    def _iter_sampled(self) -> Iterator[dict[str, Any]]:
        """Draw whole tasks, so every task carries the same weight.

        Each batch walks a shuffled task order, taking one random puzzle per
        task and as many of its augmented views as still fit. A short final
        batch is dropped: it would be a partial task rather than a partial
        epoch.
        """
        # Philox rather than the ambient stream: the sampling sequence must be
        # replayable from the seed and the pass count alone.
        rng = np.random.Generator(np.random.Philox(seed=self.seed + self.passes))
        self.passes += 1
        order = rng.permutation(self.num_tasks)
        cursor = 0
        while cursor < order.size:
            rows: list[np.ndarray] = []
            puzzle_ids: list[np.ndarray] = []
            filled = 0
            while cursor < order.size and filled < self.batch_size:
                task = int(order[cursor])
                cursor += 1
                lo, hi = int(self.groups[task]), int(self.groups[task + 1])
                if hi <= lo:
                    continue
                puzzle = int(rng.integers(lo, hi))
                start = int(self.puzzles[puzzle])
                size = int(self.puzzles[puzzle + 1]) - start
                take = min(size, self.batch_size - filled)
                rows.append(start + rng.choice(size, take, replace=False))
                puzzle_ids.append(np.full(take, puzzle, dtype=np.int64))
                filled += take
            if filled < self.batch_size:
                break  # a partial task, not a partial epoch
            yield self._batch(
                np.concatenate(rows).astype(np.int64),
                np.concatenate(puzzle_ids),
                valid=self.batch_size,
            )

    def _iter_ordered(self) -> Iterator[dict[str, Any]]:
        """Walk every row once, so pass@K sees every ballot."""
        total = int(self.puzzles[-1])
        for start in range(0, total, self.batch_size):
            end = min(total, start + self.batch_size)
            rows = np.arange(start, end, dtype=np.int64)
            # Which puzzle each row belongs to, for the per-task prefix.
            puzzle_ids = np.searchsorted(self.puzzles, rows, side="right") - 1
            yield self._batch(rows, puzzle_ids, valid=int(end - start))

    def _batch(
        self,
        rows: np.ndarray,
        puzzle_ids: np.ndarray,
        *,
        valid: int,
    ) -> dict[str, Any]:
        """Gather one batch, padding it to full width."""
        index = torch.from_numpy(rows).to(self.device)
        media = self.inputs[index]
        labels = self.labels[index]
        # The build marks skipped cells with its own id; the loss and the halt
        # target both key on -100, so remap once here rather than at each use.
        labels = torch.where(
            labels == self.ignore_label_id,
            torch.full_like(labels, -100),
            labels,
        )
        identifiers = torch.from_numpy(
            self.identifiers[puzzle_ids].astype(np.int64),
        ).to(self.device)
        if valid < self.batch_size:
            pad = self.batch_size - valid
            media = torch.cat([media, media.new_zeros(pad, media.shape[1])])
            labels = torch.cat([labels, labels.new_full((pad, labels.shape[1]), -100)])
            identifiers = torch.cat([identifiers, identifiers.new_zeros(pad)])
        return {
            "media": media,
            "label": labels,
            "valid_count": valid,
            "puzzle_identifiers": identifiers,
        }


def _load_split(dataset_dir: Path, split: str) -> dict[str, Any]:
    """Read one prepared split into tensors.

    Args:
      dataset_dir: Dataset root holding ``train/`` and ``test/``.
      split: Which one to read.

    Returns:
      data: Arrays plus the metadata the batch contract needs.

    Raises:
      FileNotFoundError: If the split or its metadata is missing.

    """
    path = Path(dataset_dir).expanduser() / split
    metadata_path = path / "dataset.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"no prepared ARC data at {path}; build it with "
            "`uv --quiet run --frozen python -m "
            "priml.baselines.arcagi1.scripts.prepare_data`.",
        )
    metadata = json.loads(metadata_path.read_text())
    logger.info("loading ARC split %r from %s", split, path)
    inputs = torch.from_numpy(np.load(path / "all__inputs.npy")).to(torch.int32)
    labels = torch.from_numpy(np.load(path / "all__labels.npy")).to(torch.int32)
    puzzles = np.load(path / "all__puzzle_indices.npy").astype(np.int64)
    groups = np.load(path / "all__group_indices.npy").astype(np.int64)
    identifiers = np.load(path / "all__puzzle_identifiers.npy").astype(np.int64)
    logger.info(
        "ARC %r: %d rows, %d puzzles, %d tasks",
        split,
        inputs.shape[0],
        len(puzzles) - 1,
        len(groups) - 1,
    )
    return {
        "inputs": inputs,
        "labels": labels,
        "puzzle_indices": puzzles,
        "group_indices": groups,
        "puzzle_identifiers": identifiers,
        "ignore_label_id": metadata.get("ignore_label_id", 0),
    }
