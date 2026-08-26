"""Sudoku puzzles, served from device memory.

The prepared dataset is a flat cross-product of (puzzle, augmentation) pairs::

    all__inputs.npy        [n_samples, 81] input tokens
    all__labels.npy        [n_samples, 81] solution tokens
    all__group_indices.npy [n_puzzles + 1] boundaries between puzzles
    dataset.json           {"vocab_size": int, "seq_len": int}

Tokens are ``0`` pad, ``1`` empty cell, ``2``-``10`` the digits 1-9. Pad exists
only so a short final batch keeps the batch shape; a real puzzle has no pad.

Every split fits in the memory of any device that can train on it, so it is
held resident and batches are index slices. That removes the host-to-device
copy and the worker processes a general loader needs, which matters because a
step here is milliseconds and would otherwise be dominated by input latency.

``scripts/prepare_data.py`` downloads and builds the arrays; this module only
reads them, so constructing a config never touches the network.

Augmentation happens here rather than in the train step because sudoku's
symmetries act on the DATA (relabel the digits, reflect the grid) and produce
another valid puzzle with a correspondingly permuted solution -- the label must
move with the input, which the train step never sees.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self, cast, override

import functools
import json
import logging

from configgle import Fig
from torch import Tensor

import numpy as np
import torch

from priml.math.basic import ceil_div
from priml.math.seed import salt
from priml.paths import resolve_working_dir
from priml.runtime import get_device
from priml.timer import CheckpointableStepTimer


logger = logging.getLogger(__name__)


def augment_sudoku(
    inputs: Tensor,
    labels: Tensor,
    *,
    vocab_size: int = 11,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Relabel digits and reflect the grid, preserving the puzzle's solution.

    Sudoku's constraints are invariant under two group actions: permuting which
    digit symbol means what, and any of the 8 symmetries of the square. Applying
    both to a puzzle and its solution together yields a different puzzle with
    the same difficulty, which is why the label must be transformed alongside
    the input.

    The empty-cell marker is NOT permuted -- it is not a digit, and mixing it
    into the permutation would silently turn empty cells into clues.

    Args:
      inputs: ``[B, 81]`` input tokens.
      labels: ``[B, 81]`` solution tokens.
      vocab_size: Token table size; the permutation covers the digits within it.
      generator: RNG for the draws. ``None`` uses the ambient global stream.

    Returns:
      inputs: ``[B, 81]`` transformed inputs.
      labels: ``[B, 81]`` solutions under the same transformation.

    """
    n = 9
    batch = inputs.shape[0]
    device = inputs.device
    # Digits occupy tokens 2..10; 0 (pad) and 1 (empty) are fixed points, so a
    # padded row passes through unchanged and stays padded.
    permutations = (
        torch.arange(vocab_size, device=device, dtype=torch.long)
        .expand(batch, -1)
        .clone()
    )
    draws = torch.rand(batch, n, device=device, generator=generator)
    permutations[:, 2 : n + 2] = draws.argsort(dim=1) + 2
    inputs = torch.gather(permutations, 1, inputs.long()).to(inputs.dtype)
    labels = torch.gather(permutations, 1, labels.long()).to(labels.dtype)

    symmetries = _square_symmetries(n, device)
    choice = torch.randint(0, 8, (batch,), device=device, generator=generator)
    selected = symmetries[choice]
    return torch.gather(inputs, 1, selected), torch.gather(labels, 1, selected)


@functools.cache
def _square_symmetries(n: int, device: torch.device) -> Tensor:
    """The 8 symmetries of an ``n x n`` grid, as flat index permutations."""
    base = torch.arange(n * n, device=device).reshape(n, n)
    out: list[Tensor] = []
    for k in range(4):
        rotated = torch.rot90(base, k)
        out.append(rotated.reshape(-1))
        out.append(rotated.flip(1).reshape(-1))
    return torch.stack(out)


class SudokuData:
    """Sudoku held in device memory, yielding ``media`` / ``label`` batches.

    Every batch is exactly ``batch_size`` rows: a short final batch is padded
    with zero rows and reports how many are real, so downstream tensor shapes
    never change mid-epoch.

    Raises:
      FileNotFoundError: If the prepared arrays are absent. Run
        ``uv --quiet run --frozen python -m
        priml.baselines.sudoku.scripts.prepare_data`` first.

    """

    class Config(Fig["SudokuData"]):
        """Where the prepared arrays live, and how batches are drawn."""

        base_dir: Path | str | None = None
        """Resource root supplied during parent finalization."""

        working_dir: Path | str = "/datasets/sudoku-extreme"
        """Directory holding the ``train/`` and ``test/`` splits.

        Resolved beneath ``base_dir`` at finalize, so it names a location
        within the resource root rather than an absolute filesystem path."""

        batch_size: int = 384
        """Puzzles per training batch."""

        eval_batch_size: int | None = None
        """Puzzles per evaluation batch; ``None`` reuses ``batch_size``."""

        device: str = "auto"
        """Device holding the resident arrays ("auto" picks the best)."""

        augment: bool = True
        """Apply digit relabeling and grid symmetry per training batch.

        The prepared data already contains many transformed copies of each
        puzzle; this adds fresh variety per batch on top, so a puzzle is
        rarely seen in exactly the same orientation twice."""

        seed: int | None = None
        """Shuffle seed. ``None`` draws from the ambient global stream; an int
        drives a named per-epoch stream independent of ambient draws."""

        augment_seed: int | None = None
        """Augmentation seed. ``None`` uses the global stream; an int drives a
        named stream independent for every epoch and batch."""

        num_train_puzzles: int | None = None
        """Training puzzles to load; ``None`` loads all of them."""

        num_eval_puzzles: int | None = None
        """Evaluation puzzles to load; ``None`` loads all of them.

        The full test split is large, so mid-training evaluation normally reads
        a fixed prefix of it as a proxy and the final number is measured on the
        whole thing. The two populations are not comparable."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {config.batch_size}.")
        if config.eval_batch_size is not None and config.eval_batch_size <= 0:
            raise ValueError(
                f"eval_batch_size must be positive; got {config.eval_batch_size}.",
            )
        self.config = config
        self.dataset_dir = Path(config.working_dir)
        self.batch_size = config.batch_size
        self.eval_batch_size = config.eval_batch_size or config.batch_size
        self.timer_epoch = CheckpointableStepTimer()
        """Passes over the training split; ticked by the loop, read by the step.

        The same count as ``_epochs`` below, kept separately because that one
        is an INPUT -- it seeds the shuffle -- while this is the record a
        budget and a schedule read."""

        # Completed epochs, persisted across resume so a restored run continues
        # the shuffle sequence instead of replaying epoch 0.
        self._epochs = 0
        self._live: _SudokuBatches | None = None
        self._pending_loader_state: dict[str, Any] | None = None

    def train_dataloader(self) -> _SudokuBatches:
        """Build the re-iterable training stream."""
        # Snapshot any prior stream's epoch first, so re-creating the loader
        # continues the sequence rather than restarting it.
        if self._live is not None:
            self._epochs = self._live.epoch
        stream = _SudokuBatches(
            dataset_dir=self.dataset_dir,
            device=self.config.device,
            batch_size=self.batch_size,
            split="train",
            shuffle=True,
            num_puzzles=self.config.num_train_puzzles,
            seed=self.config.seed,
            epoch=self._epochs,
            augment=self.config.augment,
            augment_seed=self.config.augment_seed,
        )
        if self._pending_loader_state is not None:
            stream.load_state_dict(self._pending_loader_state)
            self._pending_loader_state = None
        self._live = stream
        return stream

    def eval_dataloader(self) -> _SudokuBatches:
        """Build the evaluation stream: disk order, never augmented.

        Evaluation must score the real puzzles, so neither shuffling nor
        augmentation applies -- the score would otherwise depend on which
        transformation happened to be drawn.
        """
        return _SudokuBatches(
            dataset_dir=self.dataset_dir,
            device=self.config.device,
            batch_size=self.eval_batch_size,
            split="test",
            shuffle=False,
            num_puzzles=self.config.num_eval_puzzles,
            seed=self.config.seed,
            epoch=0,
            augment=False,
            augment_seed=None,
        )

    def state_dict(self) -> dict[str, Any]:
        """Snapshot the active epoch and its next batch."""
        loader_state = (
            self._live.state_dict()
            if self._live is not None
            else self._pending_loader_state
        )
        return {
            "epoch": self._live.epoch if self._live is not None else self._epochs,
            "loader": loader_state,
            "timer_epoch": self.timer_epoch.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""
        if "timer_epoch" in state_dict:
            self.timer_epoch.load_state_dict(state_dict["timer_epoch"])
        self._epochs = int(state_dict.get("epoch", self.timer_epoch.global_count))
        loader_state_raw = state_dict.get("loader")
        if isinstance(loader_state_raw, dict):
            loader_state = cast(dict[str, Any], loader_state_raw)
            self._pending_loader_state = loader_state
            if self._live is not None:
                self._live.load_state_dict(loader_state)
                self._pending_loader_state = None
        elif self._live is not None:
            self._live.epoch = self._epochs


class _SudokuBatches:
    """One split, resident on device, iterated in fixed-size batches."""

    def __init__(
        self,
        *,
        dataset_dir: Path,
        device: torch.device | str,
        batch_size: int,
        split: str,
        shuffle: bool,
        num_puzzles: int | None,
        seed: int | None,
        epoch: int,
        augment: bool,
        augment_seed: int | None,
    ) -> None:
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative; got {epoch}.")
        data = _load_split(dataset_dir, split)
        bounds = data["group_indices"]
        if num_puzzles is not None and num_puzzles < len(bounds) - 1:
            rows = int(bounds[num_puzzles])
            bounds = bounds[: num_puzzles + 1]
        else:
            rows = len(data["inputs"])

        self.device = get_device(device)
        self.inputs: Tensor = data["inputs"][:rows].to(self.device)
        self.labels: Tensor = data["labels"][:rows].to(self.device)
        self.bounds: Tensor = bounds.to(self.device)
        self.vocab_size = int(data["vocab_size"])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.augment = augment
        self.augment_seed = augment_seed
        self.epoch = epoch
        self._active_epoch: int | None = None
        self._next_batch = 0

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield every row once, shuffled within and across puzzles."""
        if self._active_epoch is None:
            self._active_epoch = self.epoch
            self.epoch += 1
            self._next_batch = 0
        active_epoch = self._active_epoch
        generator = self._shuffle_generator(active_epoch)
        starts = self.bounds[:-1]
        sizes = (self.bounds[1:] - starts).long()

        # Two-level shuffle: permute each puzzle's own augmented copies, then
        # permute globally. The first keeps a puzzle's variants from arriving
        # in a fixed order; the second keeps whole puzzles from doing so.
        chunks: list[Tensor] = []
        for i in range(len(sizes)):
            start, size = int(starts[i]), int(sizes[i])
            index = torch.arange(start, start + size, device=self.device)
            if self.shuffle:
                index = index[
                    torch.randperm(size, device=self.device, generator=generator)
                ]
            chunks.append(index)
        order = torch.cat(chunks)
        if self.shuffle:
            order = order[
                torch.randperm(len(order), device=self.device, generator=generator)
            ]

        for batch_index, start in enumerate(range(0, len(order), self.batch_size)):
            if batch_index < self._next_batch:
                continue
            rows = order[start : start + self.batch_size]
            valid = len(rows)
            inputs = self.inputs[rows]
            labels = self.labels[rows]
            if valid < self.batch_size:
                pad = self.batch_size - valid
                inputs = torch.cat([inputs, inputs.new_zeros(pad, inputs.shape[1])])
                labels = torch.cat([labels, labels.new_zeros(pad, labels.shape[1])])
            if self.augment:
                inputs, labels = augment_sudoku(
                    inputs,
                    labels,
                    vocab_size=self.vocab_size,
                    generator=self._augmentation_generator(
                        active_epoch,
                        batch_index,
                    ),
                )
            self._next_batch = batch_index + 1
            yield {"media": inputs, "label": labels, "valid_count": valid}
        self._active_epoch = None
        self._next_batch = 0

    def __len__(self) -> int:
        """Batches per epoch, counting a short final batch."""
        return ceil_div(int(self.bounds[-1]), self.batch_size)

    def state_dict(self) -> dict[str, Any]:
        """Return enough state to resume an unfinished epoch."""
        return {
            "epoch": self.epoch,
            "active_epoch": self._active_epoch,
            "next_batch": self._next_batch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore an unfinished epoch."""
        self.epoch = int(state_dict.get("epoch", self.epoch))
        active_epoch = state_dict.get("active_epoch")
        self._active_epoch = None if active_epoch is None else int(active_epoch)
        self._next_batch = int(state_dict.get("next_batch", 0))

    def _shuffle_generator(self, epoch: int) -> torch.Generator | None:
        """A named per-epoch generator, or the ambient stream."""
        if self.seed is None:
            return None
        generator = torch.Generator(device=self.device)
        generator.manual_seed(salt("sudoku_shuffle", self.seed, epoch))
        return generator

    def _augmentation_generator(
        self,
        epoch: int,
        batch_index: int,
    ) -> torch.Generator | None:
        """A named per-batch augmentation generator, or the ambient stream."""
        if self.augment_seed is None:
            return None
        generator = torch.Generator(device=self.device)
        generator.manual_seed(
            salt("sudoku_augmentation", self.augment_seed, epoch, batch_index),
        )
        return generator


def _load_split(dataset_dir: Path, split: str) -> dict[str, Any]:
    """Read one prepared split into tensors.

    Args:
      dataset_dir: Dataset root holding ``train/`` and ``test/``.
      split: Which one to read.

    Returns:
      data: ``inputs``, ``labels``, ``group_indices``, and ``vocab_size``.

    Raises:
      FileNotFoundError: If the split or its metadata is missing.

    """
    path = Path(dataset_dir).expanduser() / split
    metadata_path = path / "dataset.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"no prepared sudoku data at {path}; build it with "
            "`uv --quiet run --frozen python -m "
            "priml.baselines.sudoku.scripts.prepare_data`.",
        )
    metadata = json.loads(metadata_path.read_text())
    logger.info("loading sudoku split %r from %s", split, path)
    inputs = torch.from_numpy(np.load(path / "all__inputs.npy")).to(torch.int32)
    labels = torch.from_numpy(np.load(path / "all__labels.npy")).to(torch.int32)
    bounds = torch.from_numpy(np.load(path / "all__group_indices.npy")).long()
    logger.info(
        "sudoku %r: %d rows across %d puzzles",
        split,
        inputs.shape[0],
        len(bounds) - 1,
    )
    return {
        "inputs": inputs,
        "labels": labels,
        "group_indices": bounds,
        "vocab_size": metadata["vocab_size"],
    }
