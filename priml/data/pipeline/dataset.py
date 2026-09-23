"""Dataset pipeline with filtering and processing stages."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping, Sized
from dataclasses import field
from pathlib import Path
from typing import (
    Any,
    Final,
    Protocol,
    Self,
    cast,
    override,
    runtime_checkable,
)

import copy
import dataclasses
import logging

from configgle import Fig, Makeable

import torch
import torch.utils.data

from priml.custom_types import HasNormalizedWorkingDirPattern
from priml.data.custom_types import Processor, Source
from priml.data.pipeline.batching import (
    Batcher,
    Unbatcher,
)
from priml.data.pipeline.shortcircuit import ShortCircuitProcessor
from priml.paths import resolve_working_dir
from priml.runtime import global_device_mesh


logger = logging.getLogger(__name__)


# Once an image fails to decode, no further filter reasons are meaningful and
# logging each one floods the logs; this sentinel short-circuits both.
_DECODE_FAILED_REASON: Final = "CropDuringDecodeImage:decode_failed"


def add_filter_reason(
    sample: MutableMapping[str, object],
    filter: str,
    reason: str,
) -> None:
    """Add filter reason to sample.

    Args:
      sample: Mutable sample dict to mark for filtering; short-circuits on
        decode failure to avoid log spam.
      filter: Stage name that produced the rejection.
      reason: Specific condition causing rejection (appended to stage name).

    """
    filter_reasons_value = sample.get("filter_reasons", [])
    if not isinstance(filter_reasons_value, list):
        filter_reasons_value = []
    filter_reasons = [
        value
        for value in cast(list[object], filter_reasons_value)
        if isinstance(value, str)
    ]
    if _DECODE_FAILED_REASON in filter_reasons:
        return
    full_reason = f"{filter}:{reason}"
    filter_reasons.append(full_reason)
    sample["filter_reasons"] = filter_reasons
    if full_reason != _DECODE_FAILED_REASON:
        logger.debug("Filter: %s", full_reason)


def add_filter_reason_typed(
    sample: Mapping[str, object],
    filter: str,
    reason: str,
) -> None:
    """Add filter reason to sample (accepts TypedDict).

    Args:
      sample: Sample.
      filter: Filter.
      reason: Reason.

    """
    add_filter_reason(cast(MutableMapping[str, object], sample), filter, reason)


class _DummySource:
    """Dummy source that yields nothing. Used as default for DataPipeline.Config."""

    class Config(Fig["_DummySource"]): ...

    def __init__(self, config: Config): ...

    def __iter__(self) -> Iterator[dict[str, object]]:
        empty: list[dict[str, object]] = []
        return iter(empty)


class DataPipeline:
    """Orchestrate Source → Processors → Filters pipeline.

    Tracks filtering statistics and logs progress. Filter counts are stored
    in samples and aggregated when batches are yielded.
    """

    class Config(Fig["DataPipeline"]):
        source: Makeable[Source[Any]] = field(  # pyright: ignore[reportExplicitAny] -- A source yields whatever sample shape its first stage consumes.
            default_factory=_DummySource.Config,
        )
        """Data source that yields raw samples."""

        processors: list[Makeable[Processor[Any, Any]]] = field(  # pyright: ignore[reportExplicitAny] -- Stages are typed pairwise; a list of them has no expressible element type.
            default_factory=list[Makeable[Processor[Any, Any]]],  # pyright: ignore[reportExplicitAny] -- Stages are typed pairwise; a list of them has no expressible element type.
        )
        """Ordered list of processors/filters to apply."""

        filters_shortcircuit: bool = True
        """Skip already-filtered samples in subsequent processors."""

        enable_cache: bool = False
        """Materialize the whole pipeline into memory once (small datasets only).

        Hard constraint: the pipeline is run exactly once in the main process
        and the resulting samples are reused for every epoch. Any
        non-deterministic or GPU-resident stage (random augmentation, GPU
        tensors, model embeddings) is therefore frozen to a single realization
        and shared across epochs. Only enable this for fully deterministic,
        CPU-only pipelines where one fixed realization is intended.
        """

        base_dir: Path | str | None = None
        """Owner directory supplied during parent finalization."""

        working_dir: Path | str = "/"
        """Logical pipeline root inherited by the source."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            if (
                isinstance(self.source, HasNormalizedWorkingDirPattern)
                and self.source.base_dir is None
            ):
                self.source.base_dir = self.working_dir
            return super().finalize()

    def __init__(self, config: Config):
        self.config = config
        self.source = config.source.make()

        # Build processor list with optional short-circuit wrappers
        # Skip wrapping batching processors (Batcher, Unbatcher) since they
        # need to accumulate/distribute samples, not process one at a time.
        if config.filters_shortcircuit:
            # Wrap each processor to skip already-filtered samples.
            self.processors: list[Processor[Any, Any]] = []  # pyright: ignore[reportExplicitAny] -- Stages are typed pairwise; a list of them has no expressible element type.
            for p in config.processors:
                setup_p = p.make()
                # Don't wrap batching processors.
                if isinstance(setup_p, (Batcher, Unbatcher)):
                    self.processors.append(setup_p)
                else:
                    self.processors.append(
                        ShortCircuitProcessor.Config(processor=p).make(),
                    )
        else:
            self.processors = [p.make() for p in config.processors]

        self.samples_processed = 0
        self.samples_passed = 0
        self.samples_filtered = 0
        self.filter_counts = dict[str, int]()

    def __iter__(self) -> Iterator[dict[str, object]]:
        """Iterate through pipeline, yielding processed samples."""
        # Compose processors as iterator transformations.
        stream: Iterator[Any] = iter(self.source)  # pyright: ignore[reportExplicitAny] -- The stage chain is typed pairwise; the running stream has no single element type.
        for processor in self.processors:
            stream = processor(stream)

        # Track statistics and filter.
        yield from self._track_and_filter(stream)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        if isinstance(self.source, Sized):
            return len(self.source)
        # Returning 0 here would silently truncate a DataLoader; surface the
        # missing length instead so callers do not iterate an "empty" dataset.
        raise TypeError(
            f"Source {type(self.source).__name__} is not Sized; "
            "DataPipeline has no length.",
        )

    def create_loader(
        self,
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ) -> torch.utils.data.DataLoader[dict[str, object]]:
        """Create a DataLoader for training with per-worker pipeline instances.

        Args:
          num_workers: Number of worker processes for data loading.
          prefetch_factor: Number of batches to prefetch per worker (ignored if num_workers=0).

        Returns:
          loader: DataLoader that yields batches.

        """
        # Determine dataset and effective num_workers.
        if self.config.enable_cache:
            # Inspired by TFDS: Cache small datasets in memory to eliminate I/O
            # Reference: tensorflow_datasets/core/dataset_builder.py:1073-1155.
            dataset = list(self)
            effective_num_workers = 0
        elif num_workers == 0:
            dataset = _SingleWorkerDataset(self)
            effective_num_workers = 0
        else:
            dataset = _MultipleWorkerDataset(self.config)
            effective_num_workers = num_workers

        return torch.utils.data.DataLoader[dict[str, object]](
            dataset=dataset,
            batch_size=1,  # Pipeline owns batching.
            num_workers=effective_num_workers,
            prefetch_factor=None if effective_num_workers == 0 else prefetch_factor,
            collate_fn=_passthrough_collate,
        )

    # Accounting (per sample, counted once at this single sink): - ``samples_filtered``
    # counts each filtered sample exactly once. - ``filter_counts[reason]`` counts
    # (sample, reason) pairs, so a sample with K reasons adds K. Hence
    # ``sum(filter_counts.values()) >= samples_filtered``; this is per-reason
    # attribution, not double-counting of samples.
    def _track_and_filter(
        self,
        samples: Iterator[dict[str, object]],
    ) -> Iterator[dict[str, object]]:
        """Track statistics and optionally filter samples with filter reasons."""
        for sample in samples:
            self.samples_processed += 1

            # Track filter reasons if present. Deduplicate so the same reason
            # accumulated twice on one sample is not counted twice.
            filter_reasons_value = sample.get("filter_reasons", [])
            assert isinstance(filter_reasons_value, list)
            filter_reasons = [
                reason
                for reason in cast(list[object], filter_reasons_value)
                if isinstance(reason, str)
            ]
            if filter_reasons:
                self.samples_filtered += 1
                for reason in dict.fromkeys(filter_reasons):
                    self.filter_counts[reason] = self.filter_counts.get(reason, 0) + 1
                # Only skip if filters_shortcircuit is True.
                if self.config.filters_shortcircuit:
                    continue

            self.samples_passed += 1

            yield sample


def _passthrough_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    """Pipeline pre-batches, just extract from DataLoader wrapper."""
    if len(batch) != 1:
        raise ValueError("Expected len(batch) == 1.")
    return batch[0]


@runtime_checkable
class _HasSlice(Protocol):
    """A source config whose data can be partitioned via a ``worker_slice`` field."""

    worker_slice: tuple[int, int] | None


@runtime_checkable
class _HasEpochSeed(Protocol):
    """A source (or its config) whose per-epoch shuffle seed can be injected."""

    epoch_seed: int


# Single-process multi-GPU loading round-robins workers across all visible devices
# (``worker_id % device_count``) to spread GPU-resident processors. Under an initialized
# distributed run each rank already owns one device (``set_device(local_rank)``, no
# ``CUDA_VISIBLE_DEVICES`` restriction), so every worker stays on the rank's own
# ``current_device`` -- round-robin there would collide with peer ranks' devices.
def _assign_gpu_to_worker(worker_id: int) -> int:
    """Assign a CUDA device to a DataLoader worker, returning its id."""
    if not torch.cuda.is_available():
        logger.debug("Worker %s: CUDA not available, using CPU", worker_id)
        return -1

    # A fork that already touched CUDA cannot reinitialize it; tolerate this
    # (common in test environments) instead of crashing the worker.
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device_id = torch.cuda.current_device()
        else:
            device_id = worker_id % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        logger.info(
            "Worker %s: assigned to GPU %s (total GPUs: %s)",
            worker_id,
            device_id,
            torch.cuda.device_count(),
        )
        return device_id
    except RuntimeError as e:
        if "Cannot re-initialize CUDA in forked subprocess" in str(e):
            logger.warning(
                "Worker %s: CUDA was initialized before fork, cannot assign GPU "
                "(this is expected in test environments)",
                worker_id,
            )
            return -1
        raise


class _SingleWorkerDataset(torch.utils.data.IterableDataset[dict[str, object]]):
    """Thin wrapper for single-worker mode (num_workers=0)."""

    def __init__(self, pipeline: DataPipeline):
        super().__init__()
        self.pipeline = pipeline
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch so the next ``__iter__`` reshuffles.

        Args:
          epoch: Zero-based epoch index folded into the source shuffle seed.

        """
        self._epoch = epoch

    @override
    def __iter__(self) -> Iterator[dict[str, object]]:
        # Inject the epoch into the (persisted) source so a shuffling source
        # reshuffles per epoch, mirroring the per-fork injection in
        # _MultipleWorkerDataset. Sources that do not shuffle lack the field.
        source = self.pipeline.source
        if isinstance(source, _HasEpochSeed):
            source.epoch_seed = self._epoch
        yield from self.pipeline


class _MultipleWorkerDataset(torch.utils.data.IterableDataset[dict[str, object]]):
    """Dataset wrapper that creates per-worker pipeline instances.

    This class is used by DataPipeline.create_loader() to create a separate
    pipeline instance for each DataLoader worker process.
    """

    def __init__(self, config: DataPipeline.Config):
        super().__init__()
        self.config = config
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch so the next ``__iter__`` reshuffles.

        Called in the main process before ``iter(loader)`` re-forks the workers,
        so the value survives into each worker's deep-copied source config and
        every worker of the epoch shares one shuffle seed.

        Args:
          epoch: Zero-based epoch index folded into the source shuffle seed.

        """
        self._epoch = epoch

    @override
    def __iter__(self) -> Iterator[dict[str, object]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers_inner = 1
        else:
            worker_id = worker_info.id
            num_workers_inner = worker_info.num_workers

        # CRITICAL: Deep copy config to prevent workers from sharing state.
        # Without this, worker N's slice mutation affects worker N+1's config,
        # causing data duplication/corruption across workers.
        worker_config = copy.deepcopy(self.config)
        source_config = worker_config.source

        # Fold the main-process epoch into the per-worker source config so every
        # worker of this epoch shuffles with one shared seed and the order
        # changes across epochs. A self-held counter cannot survive the re-fork
        # (each epoch builds a fresh source); the epoch must be injected here,
        # mirroring the ``worker_slice`` injection below. Sources that never shuffle
        # lack the field and need no epoch.
        if isinstance(source_config, _HasEpochSeed):
            source_config.epoch_seed = self._epoch

        # Compose the distributed data-parallel shard with the local DataLoader
        # worker shard into one global partition. Without the ``dp`` factor each
        # replica would slice only by its local worker index and every rank
        # would see the identical data -- replication, not sharding.
        dp_rank, dp_world = _distributed_shard()
        global_id = dp_rank * num_workers_inner + worker_id
        global_num = dp_world * num_workers_inner

        if global_num > 1:
            # A source without a `slice` field cannot partition its data, so
            # every worker/rank would yield the full dataset -- N-fold
            # duplication. Refuse rather than silently corrupt the epoch.
            if not isinstance(source_config, _HasSlice):
                raise TypeError(
                    f"Source {type(source_config).__name__} does not support "
                    f"slicing but the global shard count is {global_num} > 1 "
                    f"(dp_world={dp_world}, num_workers={num_workers_inner}); "
                    "each shard would duplicate the full dataset. Use a single "
                    "rank with num_workers<=1 or a sliceable source.",
                )
            source_config.worker_slice = global_id, global_num

        # Pin this worker to a GPU before constructing the pipeline, so any
        # GPU-resident processor (model embedder, CUDA tensorizer) loaded during
        # construction lands on the assigned device. Gated on the pipeline
        # actually containing a CUDA processor: pure-CPU pipelines must not
        # claim a GPU context, which would needlessly initialize CUDA in every
        # fork and break CPU-only fork-safety.
        if _pipeline_uses_cuda(worker_config):
            _assign_gpu_to_worker(worker_id)

        # Create per-worker pipeline with the modified, GPU-assigned config.
        pipeline = DataPipeline(worker_config)
        yield from pipeline


# Walks the processor config tree (each ``Makeable`` is a dataclass) looking for a
# ``device`` field that resolves to ``cuda``. Pipelines without such a stage are pure-
# CPU and must not pin a GPU.
def _pipeline_uses_cuda(config: DataPipeline.Config) -> bool:
    """Return whether any processor in the pipeline targets a CUDA device."""
    seen = set[int]()
    stack: list[object] = list(config.processors)
    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        if _config_targets_cuda(node):
            return True
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            continue
        stack.extend(getattr(node, f.name) for f in dataclasses.fields(node))
    return False


# The data-parallel mesh dimension is the axis across which the dataset must be
# partitioned: each ``dp`` replica trains on a disjoint shard. Returns ``(0, 1)`` (no
# distributed sharding) when there is no mesh or no ``dp`` dimension -- single-process
# or a mesh that does not data-parallelize.
def _distributed_shard() -> tuple[int, int]:
    """Return this rank's ``(dp_rank, dp_world)`` from the global device mesh."""
    mesh = global_device_mesh()
    if mesh is None or mesh.mesh_dim_names is None or "dp" not in mesh.mesh_dim_names:
        return 0, 1
    dp = mesh["dp"]
    return dp.get_local_rank(), dp.size()


# GPU-bound processors (e.g. ``AsTensor``, ``Batcher``) expose a ``device`` field whose
# value resolves to a ``cuda`` device. This is the uniform signal used to detect a GPU-
# resident stage without instantiating the processor.
def _config_targets_cuda(config: object) -> bool:
    """Return whether a config dataclass declares a CUDA ``device`` field."""
    device = getattr(config, "device", None)
    if not isinstance(device, (str, torch.device, int)):
        return False
    return torch.device(device).type == "cuda"
