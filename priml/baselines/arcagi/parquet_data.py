"""Native study-local loader for the released CHARM ARC parquet data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    NotRequired,
    Protocol,
    Self,
    SupportsIndex,
    TypedDict,
    cast,
    override,
    runtime_checkable,
)

import math
import operator

from configgle import Fig
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from treekle import from_plain

import numpy as np
import torch

from priml.baselines.arcagi.augmentation import ArcSpec, SpatialAugmentation
from priml.baselines.arcagi.parquet import (
    ParquetSourceData,
    read_parquet_source,
)
from priml.data.color_dihedral import ColorDihedral
from priml.paths import resolve_working_dir
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from priml.train.custom_types import TrainStepProtocol


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

        world_size: int = -1
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
        rank, world_size = _distributed_coordinates(
            config.rank,
            world_size=config.world_size,
        )
        self.rank = rank
        self.world_size = world_size
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
                num_replicas=world_size,
                rank=rank,
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
                f"{self.config.batch_size} on rank {self.rank}/{self.world_size}.",
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
    color_slots[:, :] = torch.arange(9, dtype=torch.int64)
    spatial = torch.tensor([1, 0, 0], dtype=torch.int64).expand(batch_size, 3).clone()
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
        color_slots[index] = torch.tensor(
            sample["colorperm_slots"],
            dtype=torch.int64,
        )
        spatial[index] = torch.tensor(
            sample["spatial_tags"],
            dtype=torch.int64,
        )
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


def _distributed_coordinates(
    rank: int,
    *,
    world_size: int,
) -> tuple[int, int]:
    if rank == -1:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1
    if world_size <= 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size).")
    return rank, world_size


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
    if (config.rank < 0) != (config.world_size < 0):
        raise ValueError("rank and world_size must both be set or both be -1.")
    if config.num_tasks is not None and config.num_tasks < 0:
        raise ValueError("num_tasks cannot be negative.")
    if config.num_instances is not None and config.num_instances < 0:
        raise ValueError("num_instances cannot be negative.")
