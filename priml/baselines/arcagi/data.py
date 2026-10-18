"""ARC-AGI tasks from prepared arrays and augmented v2 Parquet.

The prepared dataset is a three-level hierarchy, which is what makes ARC
different from a flat dataset::

    group  -- one ARC task (a rule)
      puzzle -- one augmented VIEW of that task: a color/dihedral transform,
                plus, in a spatial-eval test split, a spatial one
        example -- one input/output pair of that view

A view's examples are the task's pairs under one transform: its demonstration
pairs in ``train``, its held-out test pairs in ``test``. So in the test split
every test input of a view shares that view's puzzle id; a metric scoring per
test input must key on the canonical input, not on the id.

On disk::

    all__inputs.npy             [n_examples, 900] input tokens
    all__labels.npy             [n_examples, 900] target tokens
    all__puzzle_indices.npy     [n_puzzles + 1]   example offsets per view
    all__group_indices.npy      [n_groups + 1]    view offsets per task
    all__puzzle_identifiers.npy [n_puzzles]       per-view identifier
    all__spatial_tags.npy       [n_puzzles, 3]    scale and offsets, if spatial
    dataset.json                shape and vocabulary metadata

Tokens are ``0`` pad, ``1`` the content-boundary EOS, and ``2``-``11`` the ten ARC
colors. Grids are padded to 30x30 because ARC grids vary in size and the model
needs one shape.

PuzzleData training samples by TASK, not by row: each batch draws a random task, then a
random view of it, then that view's pairs in random order. Sampling rows
uniformly instead would over-weight tasks that happen to have more views or
pairs, and the benchmark weights every task equally.

This module contains two ARC data paths. ``PuzzleData`` reads the prepared
array hierarchy built by ``scripts/prepare_data.py`` and samples one view per
task. ``ArcParquetData`` reads the released v2 parquet hierarchy and samples
ordered pair/view records while preserving their source identities. Both paths
preserve their configured sampling semantics and source identities.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    NotRequired,
    Protocol,
    Self,
    SupportsIndex,
    SupportsInt,
    TypedDict,
    cast,
    override,
    runtime_checkable,
)

import logging
import math
import operator

from configgle import Fig
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from treekle import from_plain, json_loads

import numpy as np
import torch
import torch.distributed as dist

from priml.baselines.arcagi.augmentation import (
    ArcAugmentation,
    ArcSpec,
    SpatialAugmentation,
)
from priml.baselines.arcagi.scripts.build_dataset import ensure_arc_dataset
from priml.data.color_dihedral import ColorDihedral
from priml.paths import resolve_working_dir
from priml.runtime import get_device
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from pyarrow.parquet import ParquetFile

    from priml.train.custom_types import TrainStepProtocol
else:
    from wrapt import lazy_import

    ParquetFile = lazy_import("pyarrow.parquet", "ParquetFile")


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
    metadata = json_loads(metadata_path.read_text(), dict[str, object])
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

    Evaluation: a linear scan. Every rank yields the same batch count; a rank
    past the end emits an empty ``valid_count=0`` batch so collectives stay
    aligned.
    """

    def __init__(
        self,
        *,
        dataset_dir: Path,
        device: torch.device | str | None,
        batch_size: int,
        rank: int,
        num_replicas: int,
        train: bool,
        seed: int,
        epochs_per_iter: int = 1,
        max_samples: int | None = None,
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
        self.ignore_label_id = from_plain(
            self.metadata.get("ignore_label_id"),
            int,
            default=0,
        )
        self.blank_identifier_id = from_plain(
            self.metadata.get("blank_identifier_id"),
            int,
            default=0,
        )
        self.n_examples = len(self.inputs)
        self.n_puzzles = int(self.puzzle_indices.size - 1)
        self.n_groups = int(self.group_indices.size - 1)
        self.iters = 0
        """Train passes started; the next pass seeds Philox with ``seed + iters + 1``."""

    def __iter__(self) -> Iterator[PuzzleData.Batch]:
        """Yield sampled training batches or the ordered evaluation scan."""
        if self.train:
            yield from self._iter_train()
        else:
            yield from self._iter_test()

    def __len__(self) -> int:
        """Return the evaluation batch count, or a training upper bound.

        Training samples ONE view per task per permutation, so a pass yields
        far fewer batches than the split has rows; the training value is the
        row count divided by the global batch, an upper bound the loop never
        reads (it consumes the iterator until exhausted).
        """
        if self.train:
            total_rows = _last(self.puzzle_indices) * self.epochs_per_iter
            return total_rows // max(1, self.global_batch_size)
        return (self.n_examples + self.global_batch_size - 1) // self.global_batch_size

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
        order = np.arange(self.n_examples, dtype=np.int64)
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
        puz_ids = np.asarray(
            np.take(self.puzzle_identifiers, puz_idx),
            dtype=np.int64,
        )
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

    Rank-sharded global batches, Philox ``seed + pass`` task sampling, and
    per-puzzle spatial tags.
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

        device: torch.device | str | None = None
        """Device receiving each batch."""

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
            self._ensure_tree(config)
            path = self.dataset_dir.expanduser() / "identifiers.json"
            actual = len(json_loads(path.read_text(), list[str]))
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
        self.train_iters = 0
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
        """Build the evaluation stream, every row once in order.

        Returns:
          stream: Ordered evaluation batches.

        """
        return PuzzleBatches(
            dataset_dir=self.dataset_dir,
            device=self.config.device,
            batch_size=self.eval_batch_size,
            rank=self.rank,
            num_replicas=self.num_replicas,
            train=False,
            seed=self.config.seed,
            max_samples=self.config.max_samples,
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

    # A dataset over another builder's trees overrides this; its tree's sentinel
    # records that builder's recipe, which the ARC-AGI-1 ensure would refuse.
    def _ensure_tree(self, config: Config) -> None:
        """Stage the ARC-AGI-1 tree at ``dataset_dir`` under ``config``'s recipe."""
        ensure_arc_dataset(
            target_dir=self.dataset_dir.expanduser(),
            augmentation=config.augmentation.make(),
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


class ParquetPair(TypedDict):
    """One ARC demonstration or test example."""

    input: list[list[int]]
    output: list[list[int]] | None


class ParquetRow(TypedDict):
    """One puzzle row from the augmented v2 format."""

    puzzle_id: str
    train: list[ParquetPair]
    test: list[ParquetPair]
    offline_augs: list[str]


@dataclass(frozen=True, slots=True, kw_only=True)
class GridStore:
    """Compact flat storage for variable-size ARC grids."""

    values: NDArray[np.uint8]
    offsets: NDArray[np.int64]
    shapes: NDArray[np.int64]

    def get(self, index: int) -> NDArray[np.uint8] | None:
        """Return one grid view, or ``None`` for an absent output.

        Args:
          index: Flat grid index in this store.

        Returns:
          grid: A view into compact bytes, or ``None`` for a reserved shape.

        """
        shape = [
            _numpy_int(cast(SupportsIndex, self.shapes[index, axis]))
            for axis in range(2)
        ]
        height, width = shape
        if height == 0 or width == 0:
            return None
        start = _numpy_int(cast(SupportsIndex, self.offsets[index]))
        return self.values[start : start + height * width].reshape(height, width)


@dataclass(frozen=True, slots=True, kw_only=True)
class ParquetSourceData:
    """Base grids and compact row offsets from one physical parquet path."""

    path: Path
    puzzle_names: tuple[str, ...]
    row_train_starts: NDArray[np.int64]
    row_train_counts: NDArray[np.int64]
    row_test_starts: NDArray[np.int64]
    row_test_counts: NDArray[np.int64]
    row_test_train_starts: NDArray[np.int64]
    row_test_train_counts: NDArray[np.int64]
    row_augmentation_starts: NDArray[np.int64]
    row_augmentation_counts: NDArray[np.int64]
    offline_aug_ids: NDArray[np.uint32]
    offline_aug_strings: tuple[str, ...]
    dih_idxs: NDArray[np.uint8]
    colorperm_slots: NDArray[np.uint8]
    train_inputs: GridStore
    train_outputs: GridStore
    test_inputs: GridStore
    test_outputs: GridStore
    test_train_indices: NDArray[np.int64]
    train_original_indices: NDArray[np.uint32]
    test_original_indices: NDArray[np.uint32]

    @property
    def num_rows(self) -> int:
        """Number of parquet puzzle rows."""
        return len(self.puzzle_names)

    @property
    def num_augmentations(self) -> int:
        """Number of stored augmentation descriptors, excluding repeats."""
        return len(self.dih_idxs)

    def augmentation_name(self, index: int) -> tuple[int, tuple[int, ...]]:
        """Return a compact semantic key for one offline view.

        Args:
          index: Flat descriptor index.

        Returns:
          view: Dihedral index and nine non-background color slots.

        """
        return (
            _numpy_int(cast(SupportsIndex, self.dih_idxs[index])),
            tuple(
                _numpy_int(cast(SupportsIndex, self.colorperm_slots[index, slot]))
                for slot in range(9)
            ),
        )

    def offline_aug_name(self, index: int) -> str:
        """Return the literal raw identity string for one descriptor."""
        aug_id = _numpy_int(cast(SupportsIndex, self.offline_aug_ids[index]))
        return self.offline_aug_strings[aug_id]

    def get_pair(
        self,
        pair_index: int,
        pair_kind: int,
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8] | None, str]:
        """Resolve one flattened demonstration or test pair to canonical grids.

        Args:
          pair_index: Flat pair index in the selected pair-kind store.
          pair_kind: Zero for demonstrations and one for test examples.

        Returns:
          pair: Canonical input, optional output, and canonical puzzle name.

        """
        if pair_kind == 0:
            row_index = _numpy_int(
                cast(
                    SupportsIndex,
                    np.searchsorted(
                        self.row_train_starts,
                        pair_index,
                        side="right",
                    )
                    - 1,
                ),
            )
            inputs, outputs = self.train_inputs, self.train_outputs
        elif pair_kind == 1:
            row_index = _numpy_int(
                cast(
                    SupportsIndex,
                    np.searchsorted(
                        self.row_test_starts,
                        pair_index,
                        side="right",
                    )
                    - 1,
                ),
            )
            inputs, outputs = self.test_inputs, self.test_outputs
        else:
            raise ValueError(f"Unknown pair kind {pair_kind}.")
        input_grid = inputs.get(pair_index)
        if input_grid is None:
            raise ValueError("An ARC input grid cannot be missing.")
        return input_grid, outputs.get(pair_index), self.puzzle_names[row_index]


def read_parquet_source(path: Path, *, max_grid_size: int = 30) -> ParquetSourceData:
    """Read base grids and row indexing without expanding augmented pairs.

    Args:
      path: Physical augmented v2 parquet file.
      max_grid_size: Maximum accepted grid height and width.

    Returns:
      source: Compact base grids, pair offsets, and offline-view descriptors.

    """
    puzzle_names: list[str] = []
    seen_names: set[str] = set()
    row_train_starts: list[int] = [0]
    row_train_counts: list[int] = []
    row_test_starts: list[int] = [0]
    row_test_counts: list[int] = []
    row_test_train_starts: list[int] = [0]
    row_test_train_counts: list[int] = []
    row_augmentation_starts: list[int] = [0]
    row_augmentation_counts: list[int] = []
    offline_aug_strings: list[str] = []
    offline_aug_string_ids: dict[str, int] = {}
    offline_aug_ids = array("I")
    if offline_aug_ids.itemsize != np.dtype(np.uint32).itemsize:
        raise RuntimeError("array('I') is not uint32-sized on this platform")
    dih_idxs = bytearray()
    colorperm_slots = bytearray()
    test_train_indices: list[int] = []
    train_original_indices: list[int] = []
    test_original_indices: list[int] = []
    train_inputs = _GridStoreBuilder()
    train_outputs = _GridStoreBuilder()
    test_inputs = _GridStoreBuilder()
    test_outputs = _GridStoreBuilder()

    for row in iter_parquet_rows(path):
        if row["puzzle_id"] in seen_names:
            raise ValueError(
                f"{path.name} contains duplicate v2 puzzle row for "
                f"{row['puzzle_id']!r}.",
            )
        seen_names.add(row["puzzle_id"])
        puzzle_names.append(row["puzzle_id"])
        row_augmentation_counts.append(len(row["offline_augs"]))
        for offline_aug in row["offline_augs"]:
            if offline_aug not in offline_aug_string_ids:
                offline_aug_string_ids[offline_aug] = len(offline_aug_strings)
                offline_aug_strings.append(offline_aug)
            offline_aug_ids.append(offline_aug_string_ids[offline_aug])
            dih_idx, slots = parse_offline_aug(offline_aug)
            dih_idxs.append(dih_idx)
            colorperm_slots.extend(slots)
        row_augmentation_starts.append(len(dih_idxs))

        for original_index, pair in enumerate(row["train"]):
            output = pair["output"]
            if not _grid_fits(pair["input"], max_grid_size) or output is None:
                continue
            if not _grid_fits(output, max_grid_size):
                continue
            train_inputs.append(pair["input"])
            train_outputs.append(output)
            train_original_indices.append(original_index)
        num_train_pairs = len(train_inputs.shapes) // 2
        row_train_counts.append(num_train_pairs - row_train_starts[-1])
        row_train_starts.append(num_train_pairs)

        for original_index, pair in enumerate(row["test"]):
            if not _grid_fits(pair["input"], max_grid_size):
                continue
            test_inputs.append(pair["input"])
            test_outputs.append(pair["output"])
            test_original_indices.append(original_index)
            if pair["output"] is None or _grid_fits(pair["output"], max_grid_size):
                test_train_indices.append(len(test_inputs.shapes) // 2 - 1)
        num_test_pairs = len(test_inputs.shapes) // 2
        row_test_counts.append(num_test_pairs - row_test_starts[-1])
        row_test_starts.append(num_test_pairs)
        row_test_train_counts.append(
            len(test_train_indices) - row_test_train_starts[-1],
        )
        row_test_train_starts.append(len(test_train_indices))

    return ParquetSourceData(
        path=path,
        puzzle_names=tuple(puzzle_names),
        row_train_starts=np.asarray(row_train_starts[:-1], dtype=np.int64),
        row_train_counts=np.asarray(row_train_counts, dtype=np.int64),
        row_test_starts=np.asarray(row_test_starts[:-1], dtype=np.int64),
        row_test_counts=np.asarray(row_test_counts, dtype=np.int64),
        row_test_train_starts=np.asarray(row_test_train_starts[:-1], dtype=np.int64),
        row_test_train_counts=np.asarray(row_test_train_counts, dtype=np.int64),
        row_augmentation_starts=np.asarray(
            row_augmentation_starts[:-1],
            dtype=np.int64,
        ),
        row_augmentation_counts=np.asarray(row_augmentation_counts, dtype=np.int64),
        offline_aug_ids=np.frombuffer(offline_aug_ids, dtype=np.uint32),
        offline_aug_strings=tuple(offline_aug_strings),
        dih_idxs=np.frombuffer(dih_idxs, dtype=np.uint8),
        colorperm_slots=np.frombuffer(colorperm_slots, dtype=np.uint8).reshape(-1, 9),
        train_inputs=train_inputs.finish(),
        train_outputs=train_outputs.finish(),
        test_inputs=test_inputs.finish(),
        test_outputs=test_outputs.finish(),
        test_train_indices=np.asarray(test_train_indices, dtype=np.int64),
        train_original_indices=np.asarray(train_original_indices, dtype=np.uint32),
        test_original_indices=np.asarray(test_original_indices, dtype=np.uint32),
    )


def iter_parquet_rows(
    path: Path,
    *,
    batch_size: int = 64,
) -> Iterator[ParquetRow]:
    """Yield validated v2 rows while bounding Arrow-to-Python conversion.

    Args:
      path: Physical augmented v2 parquet file.
      batch_size: Maximum Arrow rows converted into Python objects at once.

    Yields:
      row: One typed row preserving parquet order.

    """
    parquet = ParquetFile(path)
    required = {"format_version", "puzzle_id", "train", "test", "offline_augs"}
    missing = sorted(required - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(
            f"{path.name} is not an ARC augmented v2 parquet; missing columns: "
            f"{', '.join(missing)}.",
        )
    for batch in parquet.iter_batches(
        columns=sorted(required),
        batch_size=batch_size,
        use_threads=True,
    ):
        records = from_plain(batch.to_pylist(), list[dict[str, object]])
        for record in records:
            version = from_plain(record.get("format_version"), int)
            if version != 2:
                raise ValueError(
                    f"{path.name} row has format_version={version}; expected 2.",
                )
            puzzle_id = from_plain(record.get("puzzle_id"), str)
            train = _pairs(record.get("train"), "train")
            test = _pairs(record.get("test"), "test")
            offline_augs = from_plain(
                record.get("offline_augs"),
                list[str | None],
            )
            normalized_augs = [aug or "" for aug in offline_augs]
            if not normalized_augs or normalized_augs[0] != "":
                raise ValueError(
                    f"{path.name} puzzle {puzzle_id!r} must list the no-augmentation "
                    "entry first in offline_augs.",
                )
            for aug in normalized_augs:
                parse_offline_aug(aug)
            yield {
                "puzzle_id": puzzle_id,
                "train": train,
                "test": test,
                "offline_augs": normalized_augs,
            }


def parse_offline_aug(offline_aug: str) -> tuple[int, tuple[int, ...]]:
    """Decode one supported dihedral/color-permutation augmentation string.

    Args:
      offline_aug: Empty identity marker or a supported CHARM transform chain.

    Returns:
      view: Dihedral id and nine color slots, with the background omitted.

    """
    dih_idx = 0
    colorperm_slots = tuple(range(9))
    seen: set[str] = set()
    for part in offline_aug.split("||") if offline_aug else ():
        fields = part.split("|", maxsplit=1)
        if len(fields) != 2 or fields[0] not in {"dih", "colorperm"}:
            raise ValueError(f"Unsupported offline augmentation {offline_aug!r}.")
        name, parameter = fields
        if name in seen:
            raise ValueError(
                f"Duplicate {name!r} in offline augmentation {offline_aug!r}.",
            )
        seen.add(name)
        if name == "dih":
            if parameter not in tuple(str(index) for index in range(8)):
                raise ValueError(f"Invalid dihedral index in {offline_aug!r}.")
            dih_idx = int(parameter)
        else:
            if (
                len(parameter) != 10
                or parameter[0] != "0"
                or sorted(parameter[1:]) != list("123456789")
            ):
                raise ValueError(f"Invalid color permutation in {offline_aug!r}.")
            colorperm_slots = tuple(int(color) - 1 for color in parameter[1:])
    return dih_idx, colorperm_slots


def _pairs(value: object, name: str) -> list[ParquetPair]:
    pairs = from_plain(value, list[dict[str, object]], default=[])
    result: list[ParquetPair] = []
    for pair_value in pairs:
        grid_in = _grid(pair_value.get("input"), f"{name}.input")
        output = pair_value.get("output")
        typed_pair: ParquetPair = {
            "input": grid_in,
            "output": None if output is None else _grid(output, f"{name}.output"),
        }
        result.append(typed_pair)
    return result


def _grid(value: object, name: str) -> list[list[int]]:
    rows = from_plain(value, list[list[object]])
    result: list[list[int]] = []
    width: int | None = None
    for row in rows:
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{name} grid must be rectangular.")
        cells: list[int] = []
        for cell in row:
            if (
                isinstance(cell, bool)
                or not isinstance(cell, int)
                or cell < 0
                or cell > 9
            ):
                raise ValueError(f"{name} grid cells must be integer colors in 0..9.")
            cells.append(cell)
        result.append(cells)
    if not result or width == 0:
        raise ValueError(f"{name} grid must have positive height and width.")
    return result


class _GridStoreBuilder:
    """Append grids with lossless signed64 shapes and compact byte contents."""

    def __init__(self) -> None:
        self.values = bytearray()
        self.offsets: list[int] = [0]
        self.shapes: array[int] = array("q")

    def append(self, grid: list[list[int]] | None) -> None:
        """Append one grid, reserving a zero shape for a missing output.

        Args:
          grid: Validated grid or ``None`` for a missing output.

        """
        if grid is None:
            self.shapes.extend((0, 0))
            self.offsets.append(len(self.values))
            return
        height, width = len(grid), len(grid[0])
        self.values.extend(value for row in grid for value in row)
        self.shapes.extend((height, width))
        self.offsets.append(len(self.values))

    def finish(self) -> GridStore:
        """Freeze the compact builder as NumPy arrays.

        Returns:
          store: Flat grid bytes and compact shape/offset indexes.

        """
        return GridStore(
            values=np.frombuffer(self.values, dtype=np.uint8),
            offsets=np.asarray(self.offsets, dtype=np.int64),
            shapes=np.frombuffer(self.shapes, dtype=np.int64).reshape(-1, 2),
        )


def _grid_fits(grid: list[list[int]], maximum: int) -> bool:
    """Return whether both grid sides fit the configured square canvas."""
    return len(grid) <= maximum and len(grid[0]) <= maximum


PairSelection = Literal["train", "both"]


class PairSource:
    """Resolved file and row-selection rule for one ordered train source."""

    class Config(Fig["PairSource"]):
        """Configure one source path, pair policy, and total multiplicity."""

        path: Path | str = ""
        """Logical path to one released v2 parquet file."""

        pair_selection: PairSelection = "train"
        """Select demonstrations only or demonstrations plus trainable tests."""

        multiplicity: int = 1
        """Total number of times this source appears in the final index."""


class ArcBatch(TypedDict):
    """One padded model batch plus canonical examples for metric scoring."""

    media: Tensor
    puzzle_identifiers: Tensor
    puzzle_idxs: Tensor
    dih_idxs: Tensor
    colorperm_slots: Tensor
    spatial_tags: Tensor
    valid_count: int
    canonical_puzzle_names: list[str]
    canonical_inputs: list[NDArray[np.uint8]]
    canonical_outputs: list[NDArray[np.uint8] | None]
    input_metadata: list[dict[str, object]]
    label: NotRequired[Tensor]


class PairViewSample(TypedDict):
    """One uncollated augmented puzzle-pair view."""

    media: NDArray[np.int32]
    puzzle_identifiers: int
    puzzle_idxs: int
    dih_idxs: int
    colorperm_slots: tuple[int, ...]
    spatial_tags: tuple[int, int, int]
    canonical_puzzle_name: str
    canonical_input: NDArray[np.uint8]
    canonical_output: NDArray[np.uint8] | None
    input_metadata: dict[str, object]
    label: NotRequired[NDArray[np.int32]]


@dataclass(frozen=True, slots=True, kw_only=True)
class ArcInstanceMetadata:
    """Semantic identity for the seated view addressed by an instance id."""

    puzzle_name: str
    task_id: int
    dih_idx: int
    colorperm_slots: tuple[int, ...]


class ArcParquetData:
    """Resident compact indexes over released CHARM v2 parquet sources."""

    class Config(Fig["ArcParquetData"]):
        """Source paths, sampling, canvas, and runtime controls."""

        base_dir: Path | str | None = None
        """Resource root used to resolve every configured path."""

        train_sources: list[PairSource.Config] = field(
            default_factory=list[PairSource.Config],
        )
        """Ordered train sources with pair policy and source multiplicity."""

        eval_source: Path | str = ""
        """Parquet path whose test inputs form the evaluation stream."""

        canvas_size: int = 30
        """Side length of the square token canvas."""

        batch_size: int = 192
        """Per-rank training batch size."""

        eval_batch_size: int = 256
        """Per-rank evaluation batch size."""

        num_workers: int = 4
        """DataLoader worker count."""

        seed: int = 0
        """DistributedSampler seed; sample transforms keep source RNG ownership."""

        p_translate: float = 0.8
        """Training translation probability; evaluation always uses zero."""

        rank: int = -1
        """Distributed rank, or -1 to read the initialized process group."""

        num_replicas: int = -1
        """Distributed world size, or -1 to read the initialized process group."""

        num_tasks: int | None = None
        """Optional expected semantic puzzle count, verified at runtime."""

        num_instances: int | None = None
        """Optional expected augmented puzzle-view count, verified at runtime."""

        @override
        def finalize(self) -> Self:
            """Resolve configured paths without reading data or validating controls."""
            source_configs = from_plain(
                cast(object, self.train_sources),
                list[PairSource.Config],
            )
            self.train_sources = [source.finalize() for source in source_configs]
            for source in self.train_sources:
                source.path = resolve_working_dir(self.base_dir, source.path)
            self.eval_source = resolve_working_dir(self.base_dir, self.eval_source)
            return super().finalize()

    def __init__(
        self,
        config: Config,
        *,
        rng_for_sample: Callable[[int], np.random.RandomState] | None = None,
    ) -> None:
        _validate_runtime_config(config)
        self.config = config
        self.timer_epoch = CheckpointableStepTimer()
        self.rank, self.num_replicas = resolve_rank(
            config.rank,
            config.num_replicas,
        )
        source_data_by_path: dict[Path, ParquetSourceData] = {}
        train_sources: list[_SourceSelection] = []
        task_to_id: dict[str, int] = {}
        instance_to_id: dict[tuple[str, str], int] = {}
        instance_metadata: list[ArcInstanceMetadata | None] = [None]
        for source_config in config.train_sources:
            source_path = Path(source_config.path)
            source_data = source_data_by_path.get(source_path)
            if source_data is None:
                source_data = read_parquet_source(
                    source_path,
                    max_grid_size=config.canvas_size,
                )
                source_data_by_path[source_path] = source_data
            if source_config.pair_selection == "both":
                _validate_selected_test_targets(source_data, source_path=source_path)
            for row_idx, puzzle_name in enumerate(source_data.puzzle_names):
                pair_count = _array_int(
                    source_data.row_train_counts,
                    index=row_idx,
                )
                if source_config.pair_selection == "both":
                    pair_count += _array_int(
                        source_data.row_test_train_counts,
                        index=row_idx,
                    )
                if pair_count == 0:
                    continue
                if puzzle_name not in task_to_id:
                    task_to_id[puzzle_name] = len(task_to_id)
                task_id = task_to_id[puzzle_name]
                aug_start = _array_int(
                    source_data.row_augmentation_starts,
                    index=row_idx,
                )
                aug_count = _array_int(
                    source_data.row_augmentation_counts,
                    index=row_idx,
                )
                for aug_idx in range(aug_start, aug_start + aug_count):
                    key = (puzzle_name, source_data.offline_aug_name(aug_idx))
                    if key not in instance_to_id:
                        instance_id = len(instance_to_id) + 1
                        instance_to_id[key] = instance_id
                        dih_idx, colorperm_slots = source_data.augmentation_name(
                            aug_idx,
                        )
                        instance_metadata.append(
                            ArcInstanceMetadata(
                                puzzle_name=puzzle_name,
                                task_id=task_id,
                                dih_idx=dih_idx,
                                colorperm_slots=colorperm_slots,
                            ),
                        )
            train_sources.append(
                _SourceSelection(
                    source=source_data,
                    pair_selection=source_config.pair_selection,
                    multiplicity=source_config.multiplicity,
                ),
            )

        self.train_index = _build_train_index(
            train_sources,
            task_to_id=task_to_id,
            instance_to_id=instance_to_id,
        )
        self.eval_source = source_data_by_path.get(Path(config.eval_source))
        if self.eval_source is None:
            self.eval_source = read_parquet_source(
                Path(config.eval_source),
                max_grid_size=config.canvas_size,
            )
        for row_idx, puzzle_name in enumerate(self.eval_source.puzzle_names):
            test_count = _array_int(
                self.eval_source.row_test_counts,
                index=row_idx,
            )
            if test_count == 0:
                continue
            if puzzle_name not in task_to_id:
                task_to_id[puzzle_name] = len(task_to_id)
            aug_start = _array_int(
                self.eval_source.row_augmentation_starts,
                index=row_idx,
            )
            aug_count = _array_int(
                self.eval_source.row_augmentation_counts,
                index=row_idx,
            )
            for aug_idx in range(aug_start, aug_start + aug_count):
                key = (puzzle_name, self.eval_source.offline_aug_name(aug_idx))
                if key not in instance_to_id:
                    instance_id = len(instance_to_id) + 1
                    instance_to_id[key] = instance_id
                    dih_idx, colorperm_slots = self.eval_source.augmentation_name(
                        aug_idx,
                    )
                    instance_metadata.append(
                        ArcInstanceMetadata(
                            puzzle_name=puzzle_name,
                            task_id=task_to_id[puzzle_name],
                            dih_idx=dih_idx,
                            colorperm_slots=colorperm_slots,
                        ),
                    )
        if config.num_tasks is not None and config.num_tasks != len(task_to_id):
            raise ValueError(
                f"num_tasks expected {config.num_tasks}, found {len(task_to_id)}.",
            )
        if config.num_instances is not None and config.num_instances != len(
            instance_to_id,
        ):
            raise ValueError(
                f"num_instances expected {config.num_instances}, "
                f"found {len(instance_to_id)}.",
            )
        self.num_tasks = len(task_to_id)
        self.num_instances = len(instance_to_id)
        self.task_to_id = task_to_id
        self.instance_to_id = instance_to_id
        self.instance_metadata = tuple(instance_metadata)
        self.eval_index = _build_eval_index(
            self.eval_source,
            task_to_id=task_to_id,
            instance_to_id=instance_to_id,
        )
        self.train_dataset = ArcParquetDataset(
            sources=[selection.source for selection in train_sources],
            index=self.train_index,
            canvas_size=config.canvas_size,
            p_translate=config.p_translate,
            rng_for_sample=rng_for_sample,
        )
        self.eval_dataset = ArcParquetDataset(
            sources=[self.eval_source],
            index=self.eval_index,
            canvas_size=config.canvas_size,
            p_translate=None,
            rng_for_sample=rng_for_sample,
        )
        for dataset in (self.train_dataset, self.eval_dataset):
            dataset.sampler = DistributedSampler[int](
                dataset,
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=dataset.is_training,
                seed=config.seed,
            )

    def bind_step(self, step: TrainStepProtocol) -> None:
        """Bind compact view metadata to a compatible train step.

        Args:
          step: Train step that may consume semantic identities for pooled views.

        """
        if isinstance(step, _InstanceMetadataBinder):
            step.bind_instance_metadata(self.instance_metadata)

    def train_dataloader(self) -> DataLoader[ArcBatch]:
        """Build the shuffled rank-local expanded pair/view training loader.

        Returns:
          loader: Batches from this rank's shuffled expanded-index slice.

        """
        sampler = self.train_dataset.sampler
        if sampler is None:
            raise RuntimeError("Training dataset has no distributed sampler.")
        if len(sampler) < self.config.batch_size:
            raise ValueError(
                "CHARM training has no complete rank-local training batch: "
                f"{len(sampler)} sampled instances for batch size "
                f"{self.config.batch_size} on rank {self.rank}/{self.num_replicas}.",
            )
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_train,
            drop_last=True,
        )
        return cast(DataLoader[ArcBatch], loader)

    def eval_dataloader(self) -> DataLoader[ArcBatch]:
        """Build ordered evaluation inputs while retaining sampler padding votes.

        Returns:
          loader: Ordered batches whose media contains test inputs only. The
            distributed sampler may repeat final indices to equalize ranks;
            those repeats remain valid votes. ``valid_count`` counts local
            batch rows and does not deduplicate sampler padding.

        """
        loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.eval_batch_size,
            sampler=self.eval_dataset.sampler,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_eval,
        )
        return cast(DataLoader[ArcBatch], loader)

    def state_dict(self) -> dict[str, object]:
        """Checkpoint the epoch timer, not a mid-pass sample cursor.

        Loading this state starts the current sampler epoch at its beginning;
        TrainLoop derives the epoch order and resets the sampler at each epoch.

        Returns:
          state: Timer counters; no current batch or sample cursor is saved.

        """
        return {"timer_epoch": self.timer_epoch.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the epoch timer and restart dataset iteration at epoch start.

        Args:
          state_dict: Checkpointed mapping returned by :meth:`state_dict`.

        """
        timer_state = state_dict.get("timer_epoch")
        if not isinstance(timer_state, dict):
            raise TypeError("ArcParquetData state requires a timer_epoch mapping.")
        self.timer_epoch.load_state_dict(
            from_plain(cast(object, timer_state), dict[str, object]),
        )

    def _collate_train(self, samples: list[PairViewSample]) -> ArcBatch:
        return _collate(
            samples,
            batch_size=self.config.batch_size,
            canvas_size=self.config.canvas_size,
            include_label=True,
        )

    def _collate_eval(self, samples: list[PairViewSample]) -> ArcBatch:
        return _collate(
            samples,
            batch_size=self.config.eval_batch_size,
            canvas_size=self.config.canvas_size,
            include_label=False,
        )


class ArcParquetDataset(Dataset[PairViewSample]):
    """Lazy pair/view dataset sampled by DistributedSampler over expanded indices."""

    def __init__(
        self,
        *,
        sources: Sequence[ParquetSourceData],
        index: _SampleIndex,
        canvas_size: int,
        p_translate: float | None,
        rng_for_sample: Callable[[int], np.random.RandomState] | None,
    ) -> None:
        self.sources = sources
        self.index = index
        self.canvas_size = canvas_size
        self.p_translate = p_translate
        self.rng_for_sample = rng_for_sample
        self.colors = ColorDihedral.Config().make()
        self.spatial = SpatialAugmentation.Config(
            spec=ArcSpec(max_grid=canvas_size),
        ).make()
        self.epoch = 0
        self.sampler: DistributedSampler[int] | None = None

    @property
    def is_training(self) -> bool:
        """Whether this dataset produces paired training labels."""
        return self.p_translate is not None

    def __len__(self) -> int:
        return len(self.index.pair_indices)

    def set_epoch(self, epoch: int) -> None:
        """Set the same epoch on the expanded-index DistributedSampler."""
        self.epoch = epoch
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

    @override
    def __getitem__(self, index: int) -> PairViewSample:
        source_slot = _array_int(self.index.source_slots, index=index)
        source = self.sources[source_slot]
        pair_idx = _array_int(self.index.pair_indices, index=index)
        aug_idx = _array_int(self.index.augmentation_indices, index=index)
        task_id = _array_int(self.index.task_ids, index=index)
        instance_id = _array_int(self.index.instance_ids, index=index)
        dih_idx = _array_int(source.dih_idxs, index=aug_idx)
        color_slots = tuple(_array_ints(source.colorperm_slots, index=aug_idx))
        inp, out, puzzle_name = source.get_pair(
            pair_idx,
            pair_kind=_array_int(self.index.pair_kinds, index=index),
        )
        pair_kind = _array_int(self.index.pair_kinds, index=index)
        pair_type = "train" if pair_kind == 0 else "test"
        original_pair_indices = (
            source.train_original_indices
            if pair_kind == 0
            else source.test_original_indices
        )
        original_pair_index = _array_int(original_pair_indices, index=pair_idx)
        viewed_input = self.colors.apply_explicit(
            inp,
            tid=dih_idx,
            colorperm_slots=color_slots,
        )
        viewed_output = (
            None
            if out is None
            else self.colors.apply_explicit(
                out,
                tid=dih_idx,
                colorperm_slots=color_slots,
            )
        )
        rng = (
            np.random.RandomState()
            if self.rng_for_sample is None
            else self.rng_for_sample(index)
        )
        if self.is_training:
            if self.p_translate is None:
                raise ValueError("Expected self.p_translate is not None.")
            if viewed_output is None:
                raise ValueError(
                    f"{source.path.name} puzzle {puzzle_name!r} "
                    f"training pair {original_pair_index} requires an output grid.",
                )
            spatial_tag = self._translation_tag(
                viewed_input,
                rng=rng,
                out=viewed_output,
                p_translate=self.p_translate,
            )
            media, label = self.spatial.pack_at(
                viewed_input,
                out=viewed_output,
                tag=spatial_tag,
            )
        else:
            rng.random()
            spatial_tag = (1, 0, 0)
            media = self.spatial.pack_one_at(
                viewed_input,
                tag=spatial_tag,
            )
            label = None
        sample: PairViewSample = {
            "media": media.astype(np.int32, copy=False),
            "puzzle_identifiers": instance_id,
            "puzzle_idxs": task_id,
            "dih_idxs": dih_idx,
            "colorperm_slots": color_slots,
            "spatial_tags": spatial_tag,
            "canonical_puzzle_name": puzzle_name,
            "canonical_input": inp,
            "canonical_output": out,
            "input_metadata": {
                "source_path": str(source.path),
                "pair_index": original_pair_index,
                "offline_aug_index": aug_idx,
                "offline_aug": source.offline_aug_name(aug_idx),
                "pair_type": pair_type,
            },
        }
        if label is not None:
            sample["label"] = label.astype(np.int32, copy=False)
        return sample

    def _translation_tag(
        self,
        inp: NDArray[np.uint8],
        *,
        rng: np.random.RandomState,
        out: NDArray[np.uint8],
        p_translate: float,
    ) -> tuple[int, int, int]:
        """Draw CHARM's per-pair CanvasEOS placement from its legacy RNG."""
        translate = rng.random() < p_translate
        max_r = (
            self.canvas_size
            - max(
                operator.index(cast(SupportsIndex, inp.shape[0])),
                operator.index(cast(SupportsIndex, out.shape[0])),
            )
            - 1
        )
        max_c = (
            self.canvas_size
            - max(
                operator.index(cast(SupportsIndex, inp.shape[1])),
                operator.index(cast(SupportsIndex, out.shape[1])),
            )
            - 1
        )
        if translate and max_r > 0 and max_c > 0:
            return (
                1,
                operator.index(cast(SupportsIndex, rng.randint(0, max_r + 1))),
                operator.index(cast(SupportsIndex, rng.randint(0, max_c + 1))),
            )
        return (1, 0, 0)


@dataclass(frozen=True, slots=True, kw_only=True)
class _SampleIndex:
    """Compact expanded pair/view records; source positions occupy uint8 slots."""

    source_slots: NDArray[np.uint8]
    pair_kinds: NDArray[np.uint8]
    pair_indices: NDArray[np.int64]
    augmentation_indices: NDArray[np.int64]
    task_ids: NDArray[np.int64]
    instance_ids: NDArray[np.int64]


def _build_train_index(
    sources: Sequence[_SourceSelection],
    *,
    task_to_id: Mapping[str, int],
    instance_to_id: Mapping[tuple[str, str], int],
) -> _SampleIndex:
    pieces = [
        _source_indices(
            source_slot,
            selection=selection,
            task_to_id=task_to_id,
            instance_to_id=instance_to_id,
        )
        for source_slot, selection in enumerate(sources)
    ]
    total_size = sum(
        len(piece.pair_indices) * selection.multiplicity
        for piece, selection in zip(pieces, sources, strict=True)
    )
    result = _empty_sample_index(total_size)
    cursor = 0
    for piece in pieces:
        cursor = _copy_index(result, start=cursor, source=piece)
    for piece, selection in zip(pieces, sources, strict=True):
        for _ in range(selection.multiplicity - 1):
            cursor = _copy_index(result, start=cursor, source=piece)
    return result


def _source_indices(
    source_slot: int,
    *,
    selection: _SourceSelection,
    task_to_id: Mapping[str, int],
    instance_to_id: Mapping[tuple[str, str], int],
) -> _SampleIndex:
    source = selection.source
    sample_count = 0
    for row_idx, _ in enumerate(source.puzzle_names):
        aug_start = _array_int(source.row_augmentation_starts, index=row_idx)
        aug_count = _array_int(source.row_augmentation_counts, index=row_idx)
        pair_count = _array_int(source.row_train_counts, index=row_idx)
        if selection.pair_selection == "both":
            pair_count += _array_int(source.row_test_train_counts, index=row_idx)
        sample_count += pair_count * aug_count
    index = _empty_sample_index(sample_count)
    index.source_slots.fill(source_slot)
    cursor = 0
    for row_idx, _ in enumerate(source.puzzle_names):
        puzzle_name = source.puzzle_names[row_idx]
        train_count = _array_int(source.row_train_counts, index=row_idx)
        row_kinds = [0] * train_count
        row_pairs = list(
            range(
                _array_int(source.row_train_starts, index=row_idx),
                _array_int(source.row_train_starts, index=row_idx) + train_count,
            ),
        )
        if selection.pair_selection == "both":
            test_start = _array_int(source.row_test_train_starts, index=row_idx)
            test_count = _array_int(source.row_test_train_counts, index=row_idx)
            row_kinds.extend([1] * test_count)
            row_pairs.extend(
                _array_int(source.test_train_indices, index=test_idx)
                for test_idx in range(test_start, test_start + test_count)
            )
        pair_count = len(row_pairs)
        if pair_count == 0:
            continue
        aug_start = _array_int(source.row_augmentation_starts, index=row_idx)
        aug_count = _array_int(source.row_augmentation_counts, index=row_idx)
        length = pair_count * aug_count
        end = cursor + length
        index.pair_kinds[cursor:end] = np.tile(
            np.asarray(row_kinds, dtype=np.uint8),
            aug_count,
        )
        index.pair_indices[cursor:end] = np.tile(
            np.asarray(row_pairs, dtype=np.int64),
            aug_count,
        )
        index.augmentation_indices[cursor:end] = np.repeat(
            np.arange(aug_start, aug_start + aug_count, dtype=np.int64),
            pair_count,
        )
        index.task_ids[cursor:end].fill(task_to_id[puzzle_name])
        for view_offset, aug_idx in enumerate(range(aug_start, aug_start + aug_count)):
            view_start = cursor + view_offset * pair_count
            view_end = view_start + pair_count
            index.instance_ids[view_start:view_end].fill(
                instance_to_id[(puzzle_name, source.offline_aug_name(aug_idx))],
            )
        cursor = end
    return index


def _build_eval_index(
    source: ParquetSourceData,
    *,
    task_to_id: Mapping[str, int],
    instance_to_id: Mapping[tuple[str, str], int],
) -> _SampleIndex:
    sample_count = sum(
        _array_int(source.row_test_counts, index=row_idx)
        * _array_int(source.row_augmentation_counts, index=row_idx)
        for row_idx, _ in enumerate(source.puzzle_names)
    )
    index = _empty_sample_index(sample_count)
    index.source_slots.fill(0)
    index.pair_kinds.fill(1)
    cursor = 0
    for row_idx, puzzle_name in enumerate(source.puzzle_names):
        test_start = _array_int(source.row_test_starts, index=row_idx)
        test_count = _array_int(source.row_test_counts, index=row_idx)
        if test_count == 0:
            continue
        aug_start = _array_int(source.row_augmentation_starts, index=row_idx)
        aug_count = _array_int(source.row_augmentation_counts, index=row_idx)
        pair_count = test_count
        length = pair_count * aug_count
        end = cursor + length
        index.pair_indices[cursor:end] = np.tile(
            np.arange(test_start, test_start + test_count, dtype=np.int64),
            aug_count,
        )
        index.augmentation_indices[cursor:end] = np.repeat(
            np.arange(aug_start, aug_start + aug_count, dtype=np.int64),
            pair_count,
        )
        index.task_ids[cursor:end].fill(task_to_id[puzzle_name])
        for view_offset, aug_idx in enumerate(range(aug_start, aug_start + aug_count)):
            view_start = cursor + view_offset * pair_count
            view_end = view_start + pair_count
            index.instance_ids[view_start:view_end].fill(
                instance_to_id[(puzzle_name, source.offline_aug_name(aug_idx))],
            )
        cursor = end
    return index


def _empty_sample_index(size: int) -> _SampleIndex:
    return _SampleIndex(
        source_slots=np.empty(size, dtype=np.uint8),
        pair_kinds=np.empty(size, dtype=np.uint8),
        pair_indices=np.empty(size, dtype=np.int64),
        augmentation_indices=np.empty(size, dtype=np.int64),
        task_ids=np.empty(size, dtype=np.int64),
        instance_ids=np.empty(size, dtype=np.int64),
    )


def _copy_index(
    destination: _SampleIndex,
    *,
    start: int,
    source: _SampleIndex,
) -> int:
    end = start + len(source.pair_indices)
    destination.source_slots[start:end] = source.source_slots
    destination.pair_kinds[start:end] = source.pair_kinds
    destination.pair_indices[start:end] = source.pair_indices
    destination.augmentation_indices[start:end] = source.augmentation_indices
    destination.task_ids[start:end] = source.task_ids
    destination.instance_ids[start:end] = source.instance_ids
    return end


def _collate(
    samples: list[PairViewSample],
    *,
    batch_size: int,
    canvas_size: int,
    include_label: bool,
) -> ArcBatch:
    valid_count = len(samples)
    media = torch.zeros((batch_size, canvas_size**2), dtype=torch.int32)
    labels = (
        torch.full((batch_size, canvas_size**2), -100, dtype=torch.int32)
        if include_label
        else None
    )
    puzzle_ids = torch.zeros(batch_size, dtype=torch.int64)
    task_ids = torch.zeros(batch_size, dtype=torch.int64)
    dih_ids = torch.zeros(batch_size, dtype=torch.int64)
    color_slots = torch.zeros((batch_size, 9), dtype=torch.int64)
    color_slots[:, :] = torch.arange(9)
    spatial = torch.tensor([1, 0, 0]).expand(batch_size, 3).clone()
    for index, sample in enumerate(samples):
        media[index] = torch.from_numpy(sample["media"])
        if labels is not None:
            if "label" not in sample:
                raise ValueError("Training sample is missing its label.")
            labels[index] = torch.from_numpy(sample["label"])
            labels[index][labels[index] == 0] = -100
        puzzle_ids[index] = sample["puzzle_identifiers"]
        task_ids[index] = sample["puzzle_idxs"]
        dih_ids[index] = sample["dih_idxs"]
        color_slots[index] = torch.tensor(sample["colorperm_slots"])
        spatial[index] = torch.tensor(sample["spatial_tags"])
    batch: ArcBatch = {
        "media": media,
        "puzzle_identifiers": puzzle_ids,
        "puzzle_idxs": task_ids,
        "dih_idxs": dih_ids,
        "colorperm_slots": color_slots,
        "spatial_tags": spatial,
        "valid_count": valid_count,
        "canonical_puzzle_names": [
            sample["canonical_puzzle_name"] for sample in samples
        ],
        "canonical_inputs": [sample["canonical_input"] for sample in samples],
        "canonical_outputs": [sample["canonical_output"] for sample in samples],
        "input_metadata": [sample["input_metadata"] for sample in samples],
    }
    if labels is not None:
        batch["label"] = labels
    return batch


class _IndexedValues(Protocol):
    def __getitem__(self, index: int) -> object: ...


def _array_int(values: object, *, index: int) -> int:
    indexed = cast(_IndexedValues, values)
    return _numpy_int(cast(SupportsIndex, indexed[index]))


def _array_ints(values: object, *, index: int) -> list[int]:
    indexed = cast(_IndexedValues, values)
    row = cast(_IndexedValues, indexed[index])
    return [_numpy_int(cast(SupportsIndex, row[slot])) for slot in range(9)]


def _numpy_int(value: SupportsIndex) -> int:
    return operator.index(value)


def _validate_selected_test_targets(
    source: ParquetSourceData,
    *,
    source_path: Path,
) -> None:
    """Fail on a selected test input that cannot provide a training label."""
    for selected_offset in range(len(source.test_train_indices)):
        pair_index = _array_int(source.test_train_indices, index=selected_offset)
        if source.test_outputs.get(pair_index) is not None:
            continue
        _, _, puzzle_name = source.get_pair(pair_index, pair_kind=1)
        original_index = _array_int(source.test_original_indices, index=pair_index)
        raise ValueError(
            f"{source_path.name} puzzle {puzzle_name!r} selected test pair "
            f"{original_index} has no output grid.",
        )


@runtime_checkable
class _InstanceMetadataBinder(Protocol):
    def bind_instance_metadata(
        self,
        metadata: Sequence[ArcInstanceMetadata | None],
    ) -> None: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class _SourceSelection:
    source: ParquetSourceData
    pair_selection: PairSelection
    multiplicity: int


def _validate_runtime_config(config: ArcParquetData.Config) -> None:
    """Reject invalid source and scalar controls before opening any parquet."""
    if not config.train_sources:
        raise ValueError("At least one CHARM train source is required.")
    if len(config.train_sources) > 256:
        raise ValueError(
            "At most 256 ordered train sources fit the compact uint8 source index.",
        )
    for source in config.train_sources:
        if source.pair_selection not in ("train", "both"):
            raise ValueError("pair_selection must be 'train' or 'both'.")
        if source.multiplicity < 1:
            raise ValueError("source multiplicity must be positive.")
    if config.canvas_size > 255:
        raise ValueError("canvas_size must be at most 255.")
    if min(config.canvas_size, config.batch_size, config.eval_batch_size) <= 0:
        raise ValueError("Canvas and batch sizes must be positive.")
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if (
        math.isnan(config.p_translate)
        or config.p_translate < 0
        or config.p_translate > 1
    ):
        raise ValueError("p_translate must lie in [0, 1].")
    if (config.rank < 0) != (config.num_replicas < 0):
        raise ValueError("rank and num_replicas must both be set or both be -1.")
    if config.num_tasks is not None and config.num_tasks < 0:
        raise ValueError("num_tasks cannot be negative.")
    if config.num_instances is not None and config.num_instances < 0:
        raise ValueError("num_instances cannot be negative.")


# The builder writes int32; another builder may write narrower dtypes into the same
# layout. Not checked at runtime: readers only index, and ``_collate`` casts each slice.
def _load_int32(path: Path, *, mmap: bool = False) -> NDArray[np.int32]:
    loaded = cast(object, np.load(path, mmap_mode="r" if mmap else None))
    assert isinstance(loaded, np.ndarray)
    return loaded


def _last(values: NDArray[np.integer]) -> int:
    return int(values[-1])  # pyright: ignore[reportAny] -- numpy scalar indexing is dtype-erased.


def _at(values: NDArray[np.integer], index: int) -> int:
    return int(values[index])  # pyright: ignore[reportAny] -- numpy scalar indexing is dtype-erased.
