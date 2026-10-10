"""ARC-AGI tasks, streamed from prepared arrays.

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

Training samples by TASK, not by row: each batch draws a random task, then a
random view of it, then that view's pairs in random order. Sampling rows
uniformly instead would over-weight tasks that happen to have more views or
pairs, and the benchmark weights every task equally.

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

import logging

from configgle import Fig
from torch import Tensor
from treekle.codec import from_plain, loads

import numpy as np
import torch
import torch.distributed as dist

from priml.baselines.arcagi.augmentation import ArcAugmentation, ArcSpec
from priml.baselines.arcagi.scripts.build_dataset import ensure_arc_dataset
from priml.paths import resolve_working_dir
from priml.runtime import get_device
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from numpy.typing import NDArray


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
    metadata = from_plain(loads(metadata_path.read_text()), dict[str, object])
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
            actual = len(from_plain(loads(path.read_text()), list[str]))
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


# The builder writes int32; another builder may write narrower dtypes into the same
# layout. Not checked at runtime: readers only index, and ``_collate`` casts each slice.
def _load_int32(path: Path, *, mmap: bool = False) -> NDArray[np.int32]:
    loaded = cast(object, np.load(path, mmap_mode="r" if mmap else None))
    assert isinstance(loaded, np.ndarray)
    return cast("NDArray[np.int32]", loaded)


def _last(values: NDArray[np.integer]) -> int:
    return int(values[-1])  # pyright: ignore[reportAny] -- numpy scalar indexing is dtype-erased.


def _at(values: NDArray[np.integer], index: int) -> int:
    return int(values[index])  # pyright: ignore[reportAny] -- numpy scalar indexing is dtype-erased.
