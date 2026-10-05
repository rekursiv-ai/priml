"""ARC-AGI tasks, served from device memory or streamed from prepared arrays.

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
    all__spatial_tags.npy       [n_puzzles, 3]    scale and offsets, if spatial
    dataset.json                shape and vocabulary metadata

Tokens are ``0`` pad, ``1`` the content-boundary EOS, and ``2``-``11`` the ten ARC
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

from dataclasses import field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    NotRequired,
    Protocol,
    Self,
    SupportsInt,
    TypedDict,
    cast,
    override,
)

import itertools
import logging

from configgle import Fig
from numpy.typing import NDArray
from torch import Tensor

import numpy as np
import torch
import torch.distributed as dist

from priml.baselines.arcagi1.augmentation import ArcAugmentation, ArcSpec
from priml.baselines.arcagi1.scripts.build_dataset import ensure_arc_dataset
from priml.lib.custom_json import convert, parse
from priml.math.basic import ceil_div
from priml.math.seed import salt
from priml.paths import resolve_working_dir
from priml.runtime import get_device
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


logger = logging.getLogger(__name__)


class _SamplingRng(Protocol):
    def integers(self, low: int, high: int) -> SupportsInt: ...

    def choice(
        self,
        a: int,
        size: int,
        *,
        replace: bool,
    ) -> NDArray[np.int64]: ...


class _Split(TypedDict):
    inputs: Tensor | NDArray[np.generic]
    labels: Tensor | NDArray[np.generic]
    puzzle_indices: NDArray[np.int64]
    group_indices: NDArray[np.int64]
    puzzle_identifiers: NDArray[np.int64]
    spatial_tags: NDArray[np.int64]
    ignore_label_id: int


class _ArcBatches:
    """One prepared split, iterated in fixed-size batches."""

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
        rank: int,
        num_replicas: int,
        epochs_per_iter: int,
        device_resident: bool,
    ) -> None:
        data = _load_split(dataset_dir, split=split, mmap=not device_resident)
        groups = data["group_indices"]
        puzzles = data["puzzle_indices"]
        if num_tasks is not None and num_tasks < len(groups) - 1:
            groups = groups[: num_tasks + 1]
            puzzles = puzzles[
                : int(groups[-1])  # pyright: ignore[reportAny] -- numpy scalar indexing is dtype-erased.
                + 1
            ]
            rows = int(
                puzzles[-1],  # pyright: ignore[reportAny] -- numpy scalar indexing is dtype-erased.
            )
        else:
            rows = len(data["inputs"])

        self.device = get_device(device)
        inputs = data["inputs"][:rows]
        labels = data["labels"][:rows]
        self.inputs: Tensor | NDArray[np.generic] = (
            inputs.to(self.device) if isinstance(inputs, Tensor) else inputs
        )
        self.labels: Tensor | NDArray[np.generic] = (
            labels.to(self.device) if isinstance(labels, Tensor) else labels
        )
        self.spatial_tags: Tensor = torch.from_numpy(
            np.asarray(data["spatial_tags"][: len(puzzles) - 1], dtype=np.int64),
        ).to(self.device)
        self.groups: NDArray[np.int64] = groups
        self.puzzles: NDArray[np.int64] = puzzles
        self._group_bounds = _int_list(groups)
        self._puzzle_bounds = _int_list(puzzles)
        self.identifiers: NDArray[np.int64] = data["puzzle_identifiers"][
            : len(puzzles) - 1
        ]
        self.ignore_label_id = int(data["ignore_label_id"])
        self.batch_size = batch_size
        self.rank = rank
        self.num_replicas = num_replicas
        self.global_batch_size = batch_size * num_replicas
        self.epochs_per_iter = epochs_per_iter
        self.sample_by_task = sample_by_task
        self.seed = seed
        self.passes = passes
        self._active_pass: int | None = None
        self._next_batch = 0

    @property
    def num_tasks(self) -> int:
        """Tasks in this split."""
        return len(self.groups) - 1

    def __iter__(self) -> Iterator[dict[str, object]]:
        """Yield batches: sampled by task for training, in order for eval."""
        if self.sample_by_task:
            yield from self._iter_sampled()
        else:
            yield from self._iter_ordered()

    def __len__(self) -> int:
        """Return the batches in the active or next pass."""
        if not self.sample_by_task:
            return ceil_div(
                int(
                    self.puzzles[-1],  # pyright: ignore[reportAny] -- numpy scalar indexing is dtype-erased.
                ),
                self.global_batch_size,
            )
        pass_index = self.passes if self._active_pass is None else self._active_pass
        return sum(1 for _ in self._plan_sampled(pass_index))

    class StateDict(TypedDict):
        """Where an interrupted pass stopped; ``active_pass`` is None between passes."""

        passes: int
        active_pass: int | None
        next_batch: int

    def state_dict(self) -> StateDict:
        """Return enough state to resume an unfinished sampled pass.

        Returns:
          state: The pass counter, the pass mid-iteration, and its next batch.

        """
        return {
            "passes": self.passes,
            "active_pass": self._active_pass,
            "next_batch": self._next_batch,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore an unfinished sampled pass.

        Args:
          state_dict: State dict.

        """
        state = cast(_ArcBatches.StateDict, state_dict)
        self.passes = state.get("passes", self.passes)
        self._active_pass = state.get("active_pass")
        self._next_batch = state.get("next_batch", 0)

    # Each batch walks a shuffled task order, taking one random puzzle per task and as
    # many of its augmented views as still fit. A short final batch is dropped: it would
    # be a partial task rather than a partial epoch.
    def _iter_sampled(self) -> Iterator[dict[str, object]]:
        """Draw whole tasks, so every task carries the same weight."""
        if self._active_pass is None:
            self._active_pass = self.passes
            self.passes += 1
            self._next_batch = 0
        pass_index = self._active_pass
        for batch_index, (rows, puzzle_ids) in enumerate(
            self._plan_sampled(pass_index),
        ):
            if batch_index < self._next_batch:
                continue
            self._next_batch = batch_index + 1
            local = slice(
                self.rank * self.batch_size,
                (self.rank + 1) * self.batch_size,
            )
            yield self._batch(
                rows[local],
                puzzle_ids=puzzle_ids[local],
                valid=self.batch_size,
            )
        self._active_pass = None
        self._next_batch = 0

    def _plan_sampled(self, pass_index: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Plan complete sampled batches for one reproducible pass."""
        rng = np.random.Generator(
            np.random.Philox(seed=salt("arcagi1_task_sampling", self.seed, pass_index)),
        )
        order = _int_list(
            np.concatenate(
                [rng.permutation(self.num_tasks) for _ in range(self.epochs_per_iter)],
            ),
        )
        groups = self._group_bounds
        puzzles = self._puzzle_bounds
        cursor = 0
        while True:
            rows: list[np.ndarray] = []
            puzzle_ids: list[np.ndarray] = []
            filled = 0
            while cursor < len(order) and filled < self.global_batch_size:
                task = order[cursor]
                cursor += 1
                lo = groups[task]
                hi = groups[task + 1]
                if hi <= lo:
                    continue
                puzzle = int(rng.integers(lo, hi))
                start = puzzles[puzzle]
                size = puzzles[puzzle + 1] - start
                take = min(size, self.global_batch_size - filled)
                rows.append(start + rng.choice(size, take, replace=False))
                puzzle_ids.append(np.full(take, puzzle, dtype=np.int64))
                filled += take
            if filled < self.global_batch_size:
                return
            yield (
                np.concatenate(rows),
                np.concatenate(puzzle_ids),
            )

    def _iter_ordered(self) -> Iterator[dict[str, object]]:
        """Walk every row once, so pass@K sees every ballot."""
        total = int(
            self.puzzles[-1],  # pyright: ignore[reportAny] -- numpy scalar indexing is dtype-erased.
        )
        for start in range(0, total, self.global_batch_size):
            end = min(total, start + self.global_batch_size)
            local_start = min(start + self.rank * self.batch_size, end)
            local_end = min(start + (self.rank + 1) * self.batch_size, end)
            rows = np.arange(local_start, local_end, dtype=np.int64)
            # Which puzzle each row belongs to, for the per-task prefix.
            puzzle_ids = np.searchsorted(self.puzzles, rows, side="right") - 1
            yield self._batch(
                rows,
                puzzle_ids=puzzle_ids,
                valid=int(local_end - local_start),
            )

    def _batch(
        self,
        rows: np.ndarray,
        *,
        puzzle_ids: np.ndarray,
        valid: int,
    ) -> dict[str, object]:
        """Gather one batch, padding it to full width."""
        if isinstance(self.inputs, Tensor):
            index = torch.from_numpy(rows).to(self.device)
            media = self.inputs[index]
            assert isinstance(self.labels, Tensor)
            labels = self.labels[index]
        else:
            assert isinstance(self.labels, np.ndarray)
            media = _rows_tensor(self.inputs, rows, np.int32, self.device)
            labels = _rows_tensor(self.labels, rows, np.int32, self.device)
        spatial_tags = self.spatial_tags[torch.from_numpy(puzzle_ids).to(self.device)]
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
            spatial_tags = torch.cat(
                [spatial_tags, _identity_tags(pad, device=spatial_tags.device)],
            )
        return {
            "media": media,
            "label": labels,
            "valid_count": valid,
            "puzzle_identifiers": identifiers,
            "spatial_tags": spatial_tags,
        }


def _load_split(dataset_dir: Path, *, split: str, mmap: bool = False) -> _Split:
    """Read one prepared split into tensors, or memory-map it."""
    path = Path(dataset_dir).expanduser() / split
    metadata_path = path / "dataset.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"no prepared ARC data at {path}; build it with "
            "`uv --quiet run --frozen python -m "
            "priml.baselines.arcagi1.scripts.prepare_data`.",
        )
    metadata = parse(metadata_path.read_text(), dict[str, object])
    logger.info("loading ARC split %r from %s", split, path)
    inputs: Tensor | NDArray[np.generic]
    labels: Tensor | NDArray[np.generic]
    if mmap:
        inputs = _load_int32(path / "all__inputs.npy", mmap=True)
        labels = _load_int32(path / "all__labels.npy", mmap=True)
    else:
        inputs = torch.from_numpy(np.load(path / "all__inputs.npy")).to(torch.int32)
        labels = torch.from_numpy(np.load(path / "all__labels.npy")).to(torch.int32)
    puzzles = cast(NDArray[np.int64], np.load(path / "all__puzzle_indices.npy"))
    groups = cast(NDArray[np.int64], np.load(path / "all__group_indices.npy"))
    identifiers = cast(NDArray[np.int64], np.load(path / "all__puzzle_identifiers.npy"))
    tags_path = path / "all__spatial_tags.npy"
    # Only a spatial-eval tree writes tags; every other view is the identity.
    spatial_tags = (
        _load_int32(tags_path).astype(np.int64)
        if tags_path.is_file()
        else np.tile(np.array([1, 0, 0], dtype=np.int64), (len(identifiers), 1))
    )
    if len(spatial_tags) != len(identifiers):
        raise ValueError(
            f"{tags_path} has {len(spatial_tags)} rows but the split has "
            f"{len(identifiers)} puzzles; the build is partial or corrupt.",
        )
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
        "spatial_tags": spatial_tags,
        "ignore_label_id": convert(metadata.get("ignore_label_id"), int, default=0),
    }


class ArcData:
    """ARC tasks yielded as ``media`` / ``label`` batches.

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

        augmentation: ArcAugmentation.Config = field(
            default_factory=ArcAugmentation.Config,
        )
        """Offline recipe consumed by the preparer; never reapplied to loaded rows."""

        spec: ArcSpec = field(default_factory=ArcSpec)
        """Dataset-owned packed-grid and token vocabulary configuration."""

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

        rank: int = -1
        """Distributed rank; ``-1`` with ``num_replicas=-1`` reads torch.distributed."""

        num_replicas: int = -1
        """World size; ``-1`` with ``rank=-1`` reads torch.distributed."""

        epochs_per_iter: int = 1
        """Independent task permutations one training pass concatenates."""

        device_resident: bool = True
        """Hold the corpus on device; ``False`` memory-maps it and moves each batch."""

        @override
        def finalize(self) -> Self:
            self.augmentation.spec = self.spec
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {config.batch_size}.")
        if config.eval_batch_size is not None and config.eval_batch_size <= 0:
            raise ValueError(
                f"eval_batch_size must be positive; got {config.eval_batch_size}.",
            )
        if config.epochs_per_iter <= 0:
            raise ValueError(
                f"epochs_per_iter must be positive; got {config.epochs_per_iter}.",
            )
        self.rank, self.num_replicas = resolve_rank(config.rank, config.num_replicas)
        self.config = config
        self.dataset_dir = Path(config.working_dir)
        self.batch_size = config.batch_size
        self.eval_batch_size = config.eval_batch_size or config.batch_size
        self.timer_epoch = CheckpointableStepTimer()
        """Passes over the training split; ticked by the loop, read by the step.

        The same count as ``_passes`` below, kept separately because that one
        is an INPUT -- it seeds the sampling sequence -- while this is the
        record a budget and a schedule read."""

        # Completed passes, persisted across resume so a restored run continues
        # the sampling sequence instead of replaying the first pass.
        self._passes = 0
        self._live: _ArcBatches | None = None
        self._pending_loader_state: _ArcBatches.StateDict | None = None

    def train_dataloader(self) -> _ArcBatches:
        """Build the re-iterable training stream.

        Returns:
          stream: _ArcBatches for training (continues prior pass on recreate).

        """
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
            rank=self.rank,
            num_replicas=self.num_replicas,
            epochs_per_iter=self.config.epochs_per_iter,
            device_resident=self.config.device_resident,
        )
        if self._pending_loader_state is not None:
            stream.load_state_dict(self._pending_loader_state)
            self._pending_loader_state = None
        self._live = stream
        return stream

    def eval_dataloader(self) -> _ArcBatches:
        """Build the evaluation stream: every view of every task, in order.

        Evaluation must see every augmented view, because pass@K votes across
        them -- sampling here would discard the ballots.

        Returns:
          result: _ArcBatches for evaluation (exhaustive, non-sampled).

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
            rank=self.rank,
            num_replicas=self.num_replicas,
            epochs_per_iter=1,
            device_resident=self.config.device_resident,
        )

    class StateDict(TypedDict):
        """The pass counter, the live loader's cursor, and the pass timer.

        ``loader`` is None when no pass has started and none was pending.
        """

        passes: int
        loader: _ArcBatches.StateDict | None
        timer_epoch: NotRequired[CheckpointableStepTimer.StateDict]

    def state_dict(self) -> StateDict:
        """Snapshot the sampled pass and its next batch.

        Returns:
          state: ``"passes"``, ``"loader"`` (nested loader state), and
            ``"timer_epoch"``.

        """
        loader_state = (
            self._live.state_dict()
            if self._live is not None
            else self._pending_loader_state
        )
        return {
            "passes": self._live.passes if self._live is not None else self._passes,
            "loader": loader_state,
            "timer_epoch": self.timer_epoch.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore state produced by :meth:`state_dict`.

        Args:
          state_dict: Dict from state_dict() to restore from.

        """
        state = cast(ArcData.StateDict, state_dict)
        self._passes = state["passes"]
        loader_state = state.get("loader")
        if loader_state is not None:
            self._pending_loader_state = loader_state
            if self._live is not None:
                self._live.load_state_dict(loader_state)
                self._pending_loader_state = None
        elif self._live is not None:
            self._live.passes = self._passes
        if "timer_epoch" in state:
            self.timer_epoch.load_state_dict(state["timer_epoch"])


class PuzzleSplit(TypedDict):
    """Arrays and metadata of one prepared split."""

    inputs: NDArray[np.int32]
    labels: NDArray[np.int32]
    puzzle_indices: NDArray[np.int64]
    group_indices: NDArray[np.int64]
    puzzle_identifiers: NDArray[np.int32]
    spatial_tags: NDArray[np.int32]
    metadata: dict[str, object]


def load_puzzle_dataset(
    dataset_dir: Path,
    split: str,
    max_samples: int | None = None,
) -> PuzzleSplit:
    """Load one split's arrays and metadata.

    Inputs and labels stay memory-mapped unless ``max_samples`` is set, so many
    ranks page one shared host cache instead of each reading the whole array.

    Args:
      dataset_dir: Dataset root (parent of ``train/`` and ``test/``).
      split: ``"train"`` or ``"test"``.
      max_samples: Keep only this row prefix, trimming puzzle and group tables.

    Returns:
      split: Arrays and metadata; ``spatial_tags`` is ``[scale, pad_r, pad_c]``
        per puzzle, identity when the tree has no sidecar.

    """
    data_path = Path(dataset_dir).expanduser() / split
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")
    metadata_path = data_path / "dataset.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
    metadata = parse(metadata_path.read_text(), dict[str, object])
    logger.info("loading ARC dataset split %r from %s", split, data_path)
    inputs = _load_int32(data_path / "all__inputs.npy", mmap=True)
    labels = _load_int32(data_path / "all__labels.npy", mmap=True)
    puzzle_indices = _load_int32(data_path / "all__puzzle_indices.npy")
    group_indices = _load_int32(data_path / "all__group_indices.npy")
    puzzle_identifiers = _load_int32(data_path / "all__puzzle_identifiers.npy")
    spatial_tags_path = data_path / "all__spatial_tags.npy"
    if spatial_tags_path.is_file():
        spatial_tags = _load_int32(spatial_tags_path)
        if len(spatial_tags) != len(puzzle_identifiers):
            raise ValueError(
                f"all__spatial_tags.npy has {len(spatial_tags)} rows but the "
                f"split has {len(puzzle_identifiers)} puzzles at {data_path}; the "
                "build is partial or corrupt -- re-run data ensure.",
            )
    else:
        spatial_tags = np.tile(
            np.array([1, 0, 0], dtype=np.int32),
            (puzzle_identifiers.shape[0], 1),
        )
    if max_samples is not None:
        # A cap past the row count must not invent a boundary past the last row.
        max_samples = min(max_samples, len(inputs))
        inputs = np.array(inputs[:max_samples])
        labels = np.array(labels[:max_samples])
        puzzle_indices = puzzle_indices[
            : np.searchsorted(puzzle_indices, max_samples, side="right")
        ]
        if _last(puzzle_indices) != max_samples:
            puzzle_indices = np.append(
                puzzle_indices,
                np.array([max_samples], dtype=puzzle_indices.dtype),
            )
        n_puzzles = puzzle_indices.size - 1
        puzzle_identifiers = puzzle_identifiers[:n_puzzles]
        spatial_tags = spatial_tags[:n_puzzles]
        group_indices = group_indices[group_indices <= n_puzzles]
        if _last(group_indices) != n_puzzles:
            group_indices = np.append(
                group_indices,
                np.array([n_puzzles], dtype=group_indices.dtype),
            )
    # Train sampling draws ``rng.integers(lo, hi)`` per group, which raises when empty.
    if group_indices.size >= 2 and not np.all(np.diff(group_indices) > 0):
        raise ValueError(
            f"all__group_indices.npy has an empty group (zero puzzles) at "
            f"{data_path}; the build is partial or corrupt -- re-run data ensure.",
        )
    return {
        "inputs": inputs,
        "labels": labels,
        "puzzle_indices": np.asarray(puzzle_indices, dtype=np.int64),
        "group_indices": np.asarray(group_indices, dtype=np.int64),
        "puzzle_identifiers": np.asarray(puzzle_identifiers, dtype=np.int32),
        "spatial_tags": np.asarray(spatial_tags, dtype=np.int32),
        "metadata": metadata,
    }


class PuzzleBatches:
    """One split iterated in rank-sharded batches under the TRM reference plan.

    Training: each ``__iter__`` seeds Philox with ``seed + iters`` (then
    increments ``iters``), concatenates ``epochs_per_iter`` group permutations,
    and packs each GLOBAL batch by walking groups -- one random puzzle per
    group, then that many of its rows without replacement. A short final
    global batch is dropped; each rank keeps its slice.

    Evaluation: a linear scan (optionally over a per-puzzle or per-group
    subset). Every rank yields the same batch count; a rank past the end emits
    an empty ``valid_count=0`` batch so collectives stay aligned.
    """

    def __init__(
        self,
        *,
        dataset_dir: Path,
        device: torch.device | str,
        batch_size: int,
        rank: int,
        num_replicas: int,
        train: bool,
        seed: int,
        epochs_per_iter: int = 1,
        max_samples: int | None = None,
        max_augs_per_puzzle: int | None = None,
        max_examples_per_group: int | None = None,
        puzzle_identifier_offset: int = 0,
        puzzle_identifier_remap: NDArray[np.int64] | None = None,
    ) -> None:
        data = load_puzzle_dataset(
            dataset_dir,
            "train" if train else "test",
            max_samples=max_samples,
        )
        self.device = get_device(device)
        self.batch_size = batch_size
        self.rank = rank
        self.num_replicas = num_replicas
        self.global_batch_size = batch_size * num_replicas
        self.train = train
        self.seed = seed
        self.epochs_per_iter = epochs_per_iter
        self.metadata = data["metadata"]
        self.inputs = data["inputs"]
        self.labels = data["labels"]
        self.puzzle_indices = data["puzzle_indices"]
        self.group_indices = data["group_indices"]
        self.puzzle_identifiers = data["puzzle_identifiers"]
        self.spatial_tags = data["spatial_tags"]
        self.ignore_label_id = convert(
            self.metadata.get("ignore_label_id"),
            int,
            default=0,
        )
        self.blank_identifier_id = convert(
            self.metadata.get("blank_identifier_id"),
            int,
            default=0,
        )
        # Mixed-source hooks: an explicit remap table wins, else a flat offset moves
        # every non-blank id into this source's disjoint embedding band.
        self.puzzle_identifier_offset = puzzle_identifier_offset
        self.puzzle_identifier_remap = puzzle_identifier_remap
        self.n_examples = len(self.inputs)
        self.n_puzzles = int(self.puzzle_indices.size - 1)
        self.n_groups = int(self.group_indices.size - 1)
        self.iters = 0
        """Train passes started; the next pass seeds Philox with ``seed + iters + 1``."""
        self.eval_example_index: NDArray[np.int64] | None = None
        if not train and max_augs_per_puzzle is not None:
            self.eval_example_index = self._build_eval_subset(max_augs_per_puzzle)
            self.n_examples = int(self.eval_example_index.size)
        elif not train and max_examples_per_group is not None:
            self.eval_example_index = self._build_eval_group_subset(
                max_examples_per_group,
            )
            self.n_examples = int(self.eval_example_index.size)

    def __iter__(self) -> Iterator[PuzzleData.Batch]:
        """Yield sampled training batches or the ordered evaluation scan."""
        if self.train:
            yield from self._iter_train()
        else:
            yield from self._iter_test()

    def __len__(self) -> int:
        """Return the batch count of one iteration."""
        if self.train:
            # Row-driven: a batch packs up to a puzzle's rows per group.
            total_rows = _last(self.puzzle_indices) * self.epochs_per_iter
            return total_rows // max(1, self.global_batch_size)
        return (self.n_examples + self.global_batch_size - 1) // self.global_batch_size

    # A Philox(seed) permutation prefix per puzzle: independent of rank and world size.
    def _build_eval_subset(self, max_augs_per_puzzle: int) -> NDArray[np.int64]:
        """Keep at most ``max_augs_per_puzzle`` rows of every puzzle."""
        if max_augs_per_puzzle <= 0:
            raise ValueError(
                f"max_augs_per_puzzle must be positive, got {max_augs_per_puzzle}.",
            )
        rng = np.random.Generator(np.random.Philox(seed=self.seed))
        bounds = _int_list(self.puzzle_indices)
        kept: list[NDArray[np.int64]] = []
        for lo, hi in itertools.pairwise(bounds):
            size = hi - lo
            if size <= max_augs_per_puzzle:
                kept.append(np.arange(lo, hi, dtype=np.int64))
            else:
                pick = rng.permutation(size)[:max_augs_per_puzzle]
                pick.sort()
                kept.append(lo + pick.astype(np.int64))
        if not kept:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(kept)

    # ``group_indices`` counts puzzles, so a group's rows span
    # ``puzzle_indices[group_indices[g]] : puzzle_indices[group_indices[g + 1]]``.
    def _build_eval_group_subset(
        self,
        max_examples_per_group: int,
    ) -> NDArray[np.int64]:
        """Keep at most ``max_examples_per_group`` evenly spaced rows of every group."""
        if max_examples_per_group <= 0:
            raise ValueError(
                "max_examples_per_group must be positive, got "
                f"{max_examples_per_group}.",
            )
        puzzles = _int_list(self.puzzle_indices)
        kept: list[NDArray[np.int64]] = []
        for first, last in itertools.pairwise(_int_list(self.group_indices)):
            lo, hi = puzzles[first], puzzles[last]
            size = hi - lo
            offsets = np.linspace(
                0,
                size - 1,
                num=min(max(0, size), max_examples_per_group),
                dtype=np.int64,
            )
            kept.append(lo + offsets)
        if not kept:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(kept)

    def _iter_train(self) -> Iterator[PuzzleData.Batch]:
        self.iters += 1
        rng = np.random.Generator(np.random.Philox(seed=self.seed + self.iters))
        group_order = np.concatenate(
            [rng.permutation(self.n_groups) for _ in range(self.epochs_per_iter)],
        )
        start = 0
        while start < group_order.size:
            start, ex_idx, puz_idx = self._sample_batch(rng, group_order, start)
            if ex_idx.size == self.global_batch_size:
                local = slice(
                    self.rank * self.batch_size,
                    (self.rank + 1) * self.batch_size,
                )
                yield self._collate(
                    ex_idx[local],
                    puz_idx[local],
                    valid=self.batch_size,
                )

    def _iter_test(self) -> Iterator[PuzzleData.Batch]:
        order = (
            self.eval_example_index
            if self.eval_example_index is not None
            else np.arange(self.n_examples, dtype=np.int64)
        )
        for start in range(0, order.size, self.global_batch_size):
            end = min(order.size, start + self.global_batch_size)
            local_start = min(start + self.rank * self.batch_size, end)
            local_end = min(start + (self.rank + 1) * self.batch_size, end)
            ex_range = order[local_start:local_end]
            puz_idx = np.searchsorted(self.puzzle_indices, ex_range, side="right") - 1
            yield self._collate(ex_range, puz_idx, valid=local_end - local_start)

    def _sample_batch(
        self,
        rng: _SamplingRng,
        group_order: NDArray[np.int64],
        start_index: int,
    ) -> tuple[int, NDArray[np.int64], NDArray[np.int64]]:
        """Pack one global batch by walking ``group_order`` from ``start_index``."""
        batch: list[NDArray[np.int64]] = []
        batch_puzzle_indices: list[NDArray[np.int64]] = []
        current_size = 0
        while start_index < group_order.size and current_size < self.global_batch_size:
            group_id = _at(group_order, start_index)
            puzzle_id = int(
                rng.integers(
                    _at(self.group_indices, group_id),
                    _at(self.group_indices, group_id + 1),
                ),
            )
            start_index += 1
            puzzle_start = _at(self.puzzle_indices, puzzle_id)
            puzzle_size = _at(self.puzzle_indices, puzzle_id + 1) - puzzle_start
            append_size = min(puzzle_size, self.global_batch_size - current_size)
            batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int64))
            batch.append(
                puzzle_start + rng.choice(puzzle_size, append_size, replace=False),
            )
            current_size += append_size
        if not batch:
            return start_index, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return (
            start_index,
            np.concatenate(batch).astype(np.int64),
            np.concatenate(batch_puzzle_indices),
        )

    def _collate(
        self,
        ex_idx: NDArray[np.int64],
        puz_idx: NDArray[np.int64],
        valid: int,
    ) -> PuzzleData.Batch:
        media = self._to_device(self.inputs[ex_idx])
        labels = self._to_device(self.labels[ex_idx])
        # Loss and halt target mask on -100; inputs keep pad as a real token.
        labels = torch.where(
            labels == self.ignore_label_id,
            torch.full_like(labels, -100),
            labels,
        )
        source_ids = np.asarray(
            np.take(self.puzzle_identifiers, puz_idx),
            dtype=np.int64,
        )
        if self.puzzle_identifier_remap is not None:
            puz_ids = np.asarray(
                np.take(self.puzzle_identifier_remap, source_ids),
                dtype=np.int64,
            )
        elif self.puzzle_identifier_offset:
            blank = np.equal(source_ids, self.blank_identifier_id)
            puz_ids = np.where(
                blank,
                source_ids,
                source_ids + self.puzzle_identifier_offset,
            )
        else:
            puz_ids = source_ids
        puzzle_identifiers = torch.from_numpy(puz_ids.astype(np.int64)).to(self.device)
        spatial_tags = torch.from_numpy(
            self.spatial_tags[puz_idx].astype(np.int64),
        ).to(self.device)
        if valid < self.batch_size:
            pad = self.batch_size - valid
            media = torch.cat([media, media.new_zeros(pad, media.shape[1])])
            labels = torch.cat([labels, labels.new_full((pad, labels.shape[1]), -100)])
            puzzle_identifiers = torch.cat(
                [
                    puzzle_identifiers,
                    puzzle_identifiers.new_full((pad,), self.blank_identifier_id),
                ],
            )
            identity = torch.tensor(
                [1, 0, 0],
                dtype=spatial_tags.dtype,
                device=self.device,
            )
            spatial_tags = torch.cat([spatial_tags, identity.expand(pad, 3)])
        return {
            "media": media,
            "label": labels,
            "puzzle_identifiers": puzzle_identifiers,
            "spatial_tags": spatial_tags,
            "valid_count": valid,
        }

    def _to_device(self, batch_slice: NDArray[np.int32]) -> Tensor:
        """Materialize the sampled rows as int32 and move them to the device."""
        tensor = torch.from_numpy(np.ascontiguousarray(batch_slice, dtype=np.int32))
        if self.device.type == "cuda":
            tensor = tensor.pin_memory()  # pragma: no cover -- CUDA-only page-lock
        return tensor.to(self.device, non_blocking=True)


class PuzzleData:
    """ARC tasks served from a memory-mapped tree under the TRM reference plan.

    The reference-parity loader: rank-sharded global batches, Philox
    ``seed + pass`` task sampling, per-puzzle spatial tags, and proxy-eval caps.
    ``ArcData`` is the device-resident loader ``exp000`` uses.
    """

    class Batch(TypedDict):
        """One collated batch, padded to the per-rank batch size.

        Pad rows carry the blank puzzle id, ``-100`` labels, and the identity
        spatial tag; ``valid_count`` says how many leading rows are real.
        """

        media: Tensor
        label: Tensor
        puzzle_identifiers: Tensor
        spatial_tags: Tensor
        valid_count: int

    class Config(Fig["PuzzleData"]):
        """Tree location, recipe, batching, and evaluation subset."""

        augmentation: ArcAugmentation.Config = field(
            default_factory=ArcAugmentation.Config,
        )
        """Offline recipe the tree is built with when missing."""

        spec: ArcSpec = field(default_factory=ArcSpec)
        """Dataset-owned packed-grid and token vocabulary configuration."""

        base_dir: Path | str | None = None
        """Resource root supplied during parent finalization."""

        working_dir: Path | str = "/datasets/arc1concept-aug-1000"
        """Logical ARC dataset root containing train and test splits."""

        num_puzzle_identifiers: int = 0
        """Expected identifier count; positive builds a missing tree and verifies it.

        ``0`` touches neither, for tests over a tiny prepared tree."""

        batch_size: int = 192
        """Examples per training batch per replica."""

        eval_batch_size: int | None = None
        """Examples per evaluation batch per replica; ``None`` reuses ``batch_size``."""

        device: str = "auto"
        """Device receiving each batch ("auto" picks the best)."""

        seed: int = 0
        """Base seed for the Philox task sampling and per-puzzle eval subsets."""

        rank: int = -1
        """Distributed rank; ``-1`` with ``num_replicas=-1`` reads torch.distributed."""

        num_replicas: int = -1
        """World size; ``-1`` with ``rank=-1`` reads torch.distributed."""

        epochs_per_iter: int = 1
        """Group permutations one training iteration concatenates.

        The loop's epoch counter ticks once per iteration, so the data exposure
        cap is ``max_epochs * epochs_per_iter``."""

        max_samples: int | None = None
        """Row-prefix cap per split; a smoke-test knob."""

        eval_max_samples: int | None = None
        """Row-prefix cap on evaluation; falls back to ``max_samples``.

        A prefix keeps only the first puzzles, biasing pass@K; prefer
        ``eval_max_augs_per_puzzle`` for a representative fast evaluation."""

        eval_max_augs_per_puzzle: int | None = None
        """Keep at most this many rows of every puzzle (Philox-seeded, rank-free)."""

        eval_max_examples_per_group: int | None = None
        """Keep at most this many evenly spaced rows of every task group."""

        iters_offset: int = 0
        """Initial training pass count, for resuming a run that saved no loader state."""

        @override
        def finalize(self) -> Self:
            self.augmentation.spec = self.spec
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.epochs_per_iter <= 0:
            raise ValueError(
                f"epochs_per_iter must be positive, got {config.epochs_per_iter}.",
            )
        self.dataset_dir = Path(config.working_dir)
        if config.num_puzzle_identifiers > 0:
            ensure_arc_dataset(
                target_dir=self.dataset_dir.expanduser(),
                augmentation=config.augmentation.make(),
            )
            path = self.dataset_dir.expanduser() / "identifiers.json"
            actual = len(parse(path.read_text(), list[str]))
            if actual != config.num_puzzle_identifiers:
                raise ValueError(
                    f"expected {config.num_puzzle_identifiers} puzzle identifiers "
                    f"(config) but found {actual} at {path}; the dataset build "
                    "differs or data is partial/corrupt -- re-run data ensure or "
                    "update the config constant.",
                )
        self.config = config
        self.batch_size = config.batch_size
        self.eval_batch_size = config.eval_batch_size or config.batch_size
        self.rank, self.num_replicas = resolve_rank(config.rank, config.num_replicas)
        self.timer_epoch = CheckpointableStepTimer()
        """Passes over the data; ticked by the loop when the loader runs out."""
        self.train_iters = config.iters_offset
        """Training passes started, persisted so resume continues the sequence."""
        self._live: PuzzleBatches | None = None

    def train_dataloader(self) -> PuzzleBatches:
        """Build the training stream, continuing any prior stream's pass count.

        Returns:
          stream: Sampled training batches.

        """
        if self._live is not None:
            self.train_iters = self._live.iters
        stream = PuzzleBatches(
            dataset_dir=self.dataset_dir,
            device=self.config.device,
            batch_size=self.batch_size,
            rank=self.rank,
            num_replicas=self.num_replicas,
            train=True,
            seed=self.config.seed,
            epochs_per_iter=self.config.epochs_per_iter,
            max_samples=self.config.max_samples,
        )
        stream.iters = self.train_iters
        self._live = stream
        return stream

    def eval_dataloader(self) -> PuzzleBatches:
        """Build the evaluation stream under at most one configured cap.

        Returns:
          stream: Ordered evaluation batches.

        """
        eval_cap = self.config.eval_max_samples
        if eval_cap is None:
            eval_cap = self.config.max_samples
        proxy_caps = sum(
            cap is not None
            for cap in (
                self.config.eval_max_augs_per_puzzle,
                self.config.eval_max_examples_per_group,
            )
        )
        if proxy_caps > 0 and eval_cap is not None:
            raise ValueError(
                "a prefix cap (eval_max_samples, or the max_samples fallback) is "
                "mutually exclusive with the per-puzzle / per-group proxy caps; "
                "set only one.",
            )
        if proxy_caps > 1:
            raise ValueError(
                "eval_max_augs_per_puzzle and eval_max_examples_per_group are "
                "mutually exclusive; set only one.",
            )
        return self._eval_dataloader(
            max_samples=eval_cap,
            max_augs_per_puzzle=self.config.eval_max_augs_per_puzzle,
            max_examples_per_group=self.config.eval_max_examples_per_group,
        )

    def full_eval_dataloader(self) -> PuzzleBatches:
        """Build the uncapped evaluation stream for final reported metrics."""
        return self._eval_dataloader(
            max_samples=None,
            max_augs_per_puzzle=None,
            max_examples_per_group=None,
        )

    class StateDict(TypedDict):
        """Checkpointed pass count and epoch timer; old checkpoints may omit either."""

        train_iters: NotRequired[int]
        timer_epoch: NotRequired[CheckpointableStepTimer.StateDict]

    def state_dict(self) -> StateDict:
        """Return the pass count and epoch timer.

        Returns:
          state: ``train_iters`` (the live stream's, if any) and ``timer_epoch``.

        """
        return {
            "train_iters": self._live.iters
            if self._live is not None
            else self.train_iters,
            "timer_epoch": self.timer_epoch.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore state produced by :meth:`state_dict`.

        Args:
          state_dict: Saved pass count and epoch timer.

        """
        state = cast(PuzzleData.StateDict, state_dict)
        if "train_iters" in state:
            self.train_iters = state["train_iters"]
            if self._live is not None:
                self._live.iters = self.train_iters
        if "timer_epoch" in state:
            self.timer_epoch.load_state_dict(state["timer_epoch"])

    def _eval_dataloader(
        self,
        *,
        max_samples: int | None,
        max_augs_per_puzzle: int | None,
        max_examples_per_group: int | None,
    ) -> PuzzleBatches:
        return PuzzleBatches(
            dataset_dir=self.dataset_dir,
            device=self.config.device,
            batch_size=self.eval_batch_size,
            rank=self.rank,
            num_replicas=self.num_replicas,
            train=False,
            seed=self.config.seed,
            max_samples=max_samples,
            max_augs_per_puzzle=max_augs_per_puzzle,
            max_examples_per_group=max_examples_per_group,
        )


def resolve_rank(rank: int, num_replicas: int) -> tuple[int, int]:
    """Resolve an explicit pair, or ``(-1, -1)`` from torch.distributed.

    Args:
      rank: Rank, or ``-1``.
      num_replicas: World size, or ``-1``.

    Returns:
      rank: Resolved rank.
      num_replicas: Resolved world size.

    """
    if rank == -1 and num_replicas == -1:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1
    # A half-set pair would shard silently wrong.
    if num_replicas < 1 or rank < 0 or rank >= num_replicas:
        raise ValueError(
            "rank/num_replicas must both be the -1 auto-sentinel or satisfy "
            f"0 <= rank < num_replicas; got rank={rank}, "
            f"num_replicas={num_replicas}.",
        )
    return rank, num_replicas


# The builder writes int32; experimental builders write narrower dtypes into the same
# layout. Not checked at runtime: readers only index, and ``_collate`` casts each slice.
def _load_int32(path: Path, *, mmap: bool = False) -> NDArray[np.int32]:
    loaded = cast(object, np.load(path, mmap_mode="r" if mmap else None))
    assert isinstance(loaded, np.ndarray)
    return cast("NDArray[np.int32]", loaded)


def _rows_tensor(
    values: NDArray[np.generic],
    rows: np.ndarray,
    dtype: type[np.integer],
    device: torch.device,
) -> Tensor:
    """Copy the selected rows of a memory-mapped array onto ``device``."""
    return torch.from_numpy(np.asarray(values[rows], dtype=dtype).copy()).to(device)


def _identity_tags(rows: int, *, device: torch.device) -> Tensor:
    """Return ``rows`` identity spatial tags, ``(scale 1, no offset)``."""
    tags = torch.zeros((rows, 3), dtype=torch.int64, device=device)
    tags[:, 0] = 1
    return tags


def _last(values: NDArray[np.integer]) -> int:
    return int(values[-1])  # pyright: ignore[reportAny] -- numpy scalar indexing is dtype-erased.


def _at(values: NDArray[np.integer], index: int) -> int:
    return int(values[index])  # pyright: ignore[reportAny] -- numpy scalar indexing is dtype-erased.


def _int_list(values: NDArray[np.integer]) -> list[int]:
    return convert(cast(object, values.tolist()), list[int])
