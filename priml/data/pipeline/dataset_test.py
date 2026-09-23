"""Tests for DataPipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast
from unittest.mock import Mock, patch

import multiprocessing as mp
import tempfile

from configgle import Fig

import pytest
import torch.utils.data

from priml.data.pipeline.dataset import (
    DataPipeline,
    _DummySource,
    _MultipleWorkerDataset,
    _SingleWorkerDataset,
    add_filter_reason,
    add_filter_reason_typed,
)
from priml.data.processors.custom_types import Sample
from priml.data.sources.sharding import shard_and_shuffle


if TYPE_CHECKING:
    from collections.abc import Iterator


_WORKER_CONTEXT = "fork" if "fork" in mp.get_all_start_methods() else None


def _add_filter_reason(sample: Sample, reason: str) -> None:
    """Add filter reason to sample (type-safe helper)."""
    sample.setdefault("filter_reasons", []).append(reason)


class BatchSample(TypedDict):
    """Batch of samples for testing."""

    samples: list[Sample]
    _filter_counts: dict[str, int]


class DummySource:
    """Simple source that yields test samples."""

    class Config(Fig["DummySource"]):
        num_samples: int = 10

    def __init__(self, config: Config):
        self.num_samples = config.num_samples

    def __iter__(self):
        for i in range(self.num_samples):
            yield {
                "key": f"sample_{i}",
                "width": 512,
                "height": 512,
                "caption": f"Caption {i}",
                "url": f"https://example.com/image_{i}.jpg",
            }

    def __len__(self):
        return self.num_samples


class DummyFilter:
    """Filter that rejects every other sample."""

    class Config(Fig["DummyFilter"]):
        shortcircuit: bool = False

    def __init__(self, config: Config):
        self.shortcircuit_val = config.shortcircuit
        self.count = 0

    def __call__(self, samples: Iterator[Sample]) -> Iterator[Sample]:
        for sample in samples:
            self.count += 1
            if self.count % 2 == 0:
                _add_filter_reason(sample, "DummyFilter")
            yield sample

    @property
    def shortcircuit(self) -> bool:
        return self.shortcircuit_val


class DummyProcessor:
    """Processor that adds a field."""

    class Config(Fig["DummyProcessor"]): ...

    def __init__(self, config: Config): ...

    def __call__(self, samples: Iterator[Sample]) -> Iterator[Sample]:
        for sample in samples:
            yield cast(Sample, {**sample, "processed": True})


class BatchingProcessor:
    """Processor that batches samples."""

    class Config(Fig["BatchingProcessor"]):
        size: int = 2

    def __init__(self, config: Config):
        self.size = config.size
        self.queue: list[Sample] = []

    def __call__(self, samples: Iterator[Sample]) -> Iterator[BatchSample]:
        for sample in samples:
            self.queue.append(sample)
            if len(self.queue) >= self.size:
                batch: BatchSample = {"samples": self.queue, "_filter_counts": {}}
                self.queue = []
                yield batch

        # Flush remaining samples.
        if self.queue:
            batch = {"samples": self.queue, "_filter_counts": {}}
            self.queue = []
            yield batch


class SliceableSource:
    """Source that supports slicing."""

    class Config(Fig["SliceableSource"]):
        num_samples: int = 20
        data_dir: str = ""
        worker_slice: tuple[int, int] | None = None

    def __init__(self, config: Config):
        self.num_samples = config.num_samples
        self.worker_slice = config.worker_slice

    def __iter__(self):
        start = 0
        step = 1
        if self.worker_slice is not None:
            worker_id, num_workers = self.worker_slice
            start = worker_id
            step = num_workers

        for i in range(start, self.num_samples, step):
            yield {
                "key": f"sample_{i}",
                "width": 512,
                "height": 512,
            }

    def __len__(self):
        return self.num_samples


class ShufflingSliceableSource:
    """Sliceable source that shuffles before slicing with a per-epoch seed.

    Mirrors the real sources (parquet/imagenet): shuffle-before-slice with a
    shared ``epoch_seed`` so every worker of one epoch produces the same
    permutation and the slices partition the data exactly.
    """

    class Config(Fig["ShufflingSliceableSource"]):
        num_samples: int = 20
        worker_slice: tuple[int, int] | None = None
        epoch_seed: int = 0

    def __init__(self, config: Config):
        self.num_samples = config.num_samples
        self.worker_slice = config.worker_slice
        self.epoch_seed = config.epoch_seed

    def __iter__(self):
        items = shard_and_shuffle(
            list(range(self.num_samples)),
            worker_slice=self.worker_slice,
            shuffle=True,
            epoch_seed=self.epoch_seed,
        )
        for i in items:
            yield {"key": f"sample_{i}", "width": 512, "height": 512}

    def __len__(self):
        return self.num_samples


def _worker_shard_order(
    config: DataPipeline.Config,
    epoch: int,
    worker_id: int,
    num_workers: int,
) -> list[int]:
    """Sample indices a single worker emits for ``epoch`` via set_epoch+iter."""
    dataset = _MultipleWorkerDataset(config)
    dataset.set_epoch(epoch)

    worker_info = Mock()
    worker_info.id = worker_id
    worker_info.num_workers = num_workers
    with patch("torch.utils.data.get_worker_info", return_value=worker_info):
        return [int(str(s["key"]).removeprefix("sample_")) for s in dataset]


def test_multiworker_reshuffles_across_epochs() -> None:
    """#326: a multi-worker loader reshuffles each epoch via set_epoch.

    The per-fork source cannot hold epoch state (every epoch re-forks a fresh
    source). The epoch must originate in the main process via set_epoch and be
    folded into the per-worker shuffle seed. Asserts: order differs across
    epochs, identical across workers within an epoch, and each epoch's two
    worker slices partition the dataset exactly once.
    """
    num_samples, num_workers = 24, 2
    source = ShufflingSliceableSource.Config(num_samples=num_samples)
    config = DataPipeline.Config(source=source)

    def epoch_order(epoch: int) -> list[int]:
        union: list[int] = []
        for w in range(num_workers):
            union.extend(_worker_shard_order(config, epoch, w, num_workers))
        return union

    order_0 = epoch_order(0)
    order_1 = epoch_order(1)

    # Reshuffled: epoch 0 and epoch 1 differ in order.
    assert order_0 != order_1
    # Both are permutations of the same full set.
    assert sorted(order_0) == sorted(order_1) == list(range(num_samples))

    # Within one epoch the two workers' slices partition the dataset exactly
    # once (union == full set, disjoint, no gap/dup).
    w0 = _worker_shard_order(config, 0, 0, num_workers)
    w1 = _worker_shard_order(config, 0, 1, num_workers)
    assert set(w0).isdisjoint(w1)
    assert sorted(w0 + w1) == list(range(num_samples))


def test_single_worker_reshuffles_across_epochs() -> None:
    """#326: num_workers=0 path reshuffles each epoch via set_epoch (no regress)."""
    source = ShufflingSliceableSource.Config(num_samples=20)
    pipeline = DataPipeline(DataPipeline.Config(source=source))
    dataset = _SingleWorkerDataset(pipeline)

    dataset.set_epoch(0)
    order_0 = [int(str(s["key"]).removeprefix("sample_")) for s in dataset]
    dataset.set_epoch(1)
    order_1 = [int(str(s["key"]).removeprefix("sample_")) for s in dataset]

    assert order_0 != order_1
    assert sorted(order_0) == sorted(order_1) == list(range(20))


class SliceableSourceWithBadPath:
    """Source with slice support but non-existent data_dir."""

    class Config(Fig["SliceableSourceWithBadPath"]):
        num_samples: int = 10
        data_dir: str = "/nonexistent/path/does/not/exist"
        worker_slice: tuple[int, int] | None = None

    def __init__(self, config: Config):
        self.num_samples = config.num_samples
        self.worker_slice = config.worker_slice

    def __iter__(self):
        start = 0
        step = 1
        if self.worker_slice is not None:
            worker_id, num_workers = self.worker_slice
            start = worker_id
            step = num_workers

        for i in range(start, self.num_samples, step):
            yield {"key": f"sample_{i}"}


def test_pipeline_no_filters_no_processors():
    """Test pipeline with only source."""
    source = DummySource.Config()
    source.num_samples = 5
    config = DataPipeline.Config(source=source)
    pipeline = DataPipeline(config)

    samples = list(pipeline)

    # Should yield 5 samples.
    assert len(samples) == 5
    assert all("key" in s for s in samples)


def test_pipeline_with_filter():
    """Test pipeline with filter."""
    source = DummySource.Config()
    source.num_samples = 10
    filt_config = DummyFilter.Config()
    config = DataPipeline.Config(source=source)
    config.processors = [filt_config]
    pipeline = DataPipeline(config)

    samples = list(pipeline)

    # Filter rejects every other sample, so expect 5.
    assert len(samples) == 5


def test_pipeline_filter_statistics():
    """Test filter statistics tracking."""
    source = DummySource.Config()
    source.num_samples = 10
    filt = DummyFilter.Config()
    config = DataPipeline.Config(source=source)
    config.processors = [filt]
    pipeline = DataPipeline(config)

    list(pipeline)  # Consume all.

    assert pipeline.samples_processed == 10
    assert pipeline.samples_filtered == 5
    assert pipeline.samples_passed == 5
    assert pipeline.filter_counts["DummyFilter"] == 5


def test_pipeline_with_processor():
    """Test pipeline with processor."""
    source = DummySource.Config()
    source.num_samples = 5
    processor = DummyProcessor.Config()
    config = DataPipeline.Config(source=source)
    config.processors = [processor]
    pipeline = DataPipeline(config)

    samples = list(pipeline)

    assert len(samples) == 5
    assert all(s.get("processed") is True for s in samples)


def test_pipeline_with_batcher():
    """Test pipeline with batching processor."""
    source = DummySource.Config()
    source.num_samples = 10
    batcher = BatchingProcessor.Config()
    batcher.size = 3
    config = DataPipeline.Config(source=source)
    config.processors = [batcher]
    pipeline = DataPipeline(config)

    batches = list(pipeline)

    # 10 samples, size=3 -> 3 full batches + 1 partial (when flushed)
    # But our simple batcher only yields on full, so 3 batches
    # Actually, our iterator doesn't flush, so we get 3 batches of 3.
    assert len(batches) >= 3


def test_pipeline_filter_shortcircuit():
    """Test shortcircuit behavior."""
    source = DummySource.Config()
    source.num_samples = 10

    class AlwaysRejectFilter:
        class Config(Fig["AlwaysRejectFilter"]): ...

        shortcircuit = True

        def __init__(self, config: Config): ...

        def __call__(self, samples: Iterator[Sample]) -> Iterator[Sample]:
            for sample in samples:
                _add_filter_reason(sample, "AlwaysReject")
                yield sample

    class NeverRejectFilter:
        class Config(Fig["NeverRejectFilter"]): ...

        shortcircuit = False

        def __init__(self, config: Config): ...

        def __call__(self, samples: Iterator[Sample]) -> Iterator[Sample]:
            yield from samples

    filt1 = AlwaysRejectFilter.Config()
    filt2 = NeverRejectFilter.Config()

    config = DataPipeline.Config(source=source)
    config.processors = [filt1, filt2]
    pipeline = DataPipeline(config)
    samples = list(pipeline)

    # All samples rejected by first filter (shortcircuit stops early)
    assert len(samples) == 0
    assert pipeline.samples_filtered == 10


def test_pipeline_filter_stats_in_sample():
    """Test filter stats stored in samples."""
    source = DummySource.Config()
    source.num_samples = 10
    filt = DummyFilter.Config()
    config = DataPipeline.Config(source=source)
    config.processors = [filt]
    pipeline = DataPipeline(config)

    samples: list[dict[str, object]] = list(pipeline)

    # Passed samples have empty filter reasons (no filters triggered)
    # Note: samples without filter_reasons are treated as having an empty list.
    for sample in samples:
        reasons = sample.get("filter_reasons", [])
        assert isinstance(reasons, list)
        assert len(cast(list[object], reasons)) == 0


def test_pipeline_len():
    """Test pipeline length."""
    source = DummySource.Config()
    source.num_samples = 100
    config = DataPipeline.Config(source=source)
    pipeline = DataPipeline(config)

    assert len(pipeline) == 100


def test_pipeline_multiple_filters():
    """Test pipeline with multiple filters."""
    source = DummySource.Config()
    source.num_samples = 20

    class Filter1:
        class Config(Fig["Filter1"]): ...

        shortcircuit = False

        def __init__(self, config: Config): ...

        def __call__(self, samples: Iterator[Sample]) -> Iterator[Sample]:
            for sample in samples:
                # Reject if key ends in 0.
                if sample.get("key", "").endswith("0"):
                    _add_filter_reason(sample, "Filter1")
                yield sample

    class Filter2:
        class Config(Fig["Filter2"]): ...

        shortcircuit = False

        def __init__(self, config: Config): ...

        def __call__(self, samples: Iterator[Sample]) -> Iterator[Sample]:
            for sample in samples:
                # Reject if key ends in 5.
                if sample.get("key", "").endswith("5"):
                    _add_filter_reason(sample, "Filter2")
                yield sample

    config = DataPipeline.Config(source=source)
    config.processors = [Filter1.Config(), Filter2.Config()]
    pipeline = DataPipeline(config)
    samples = list(pipeline)

    # Should reject samples 0, 5, 10, 15 -> 16 remaining.
    assert len(samples) == 16
    assert pipeline.filter_counts.get("Filter1", 0) == 2  # 0, 10.
    assert pipeline.filter_counts.get("Filter2", 0) == 2  # 5, 15.


def test_source_with_existingfilter_reasons():
    """Test that pipeline preserves existing filter_reasons from source."""

    class SourceWithFilterReasons:
        """Source that yields samples with pre-existing filter reasons."""

        class Config(Fig["SourceWithFilterReasons"]):
            num_samples: int = 5

        def __init__(self, config: Config):
            self.num_samples = config.num_samples

        def __iter__(self):
            for i in range(self.num_samples):
                sample = {
                    "key": f"sample_{i}",
                    "filter_reasons": ["SourceFilter"] if i % 2 == 0 else [],
                }
                yield sample

    source = SourceWithFilterReasons.Config()
    source.num_samples = 10
    config = DataPipeline.Config(source=source)
    pipeline = DataPipeline(config)

    samples = list(pipeline)

    # Source filtered samples 0, 2, 4, 6, 8 -> 5 remaining.
    assert len(samples) == 5
    assert pipeline.samples_filtered == 5
    assert pipeline.filter_counts.get("SourceFilter", 0) == 5


def test_dummy_source():
    """Test _DummySource returns empty iterator."""
    cfg = _DummySource.Config()
    source = cfg.make()

    samples = list(source)
    assert len(samples) == 0


def test_create_loader_no_workers():
    """Test create_loader with num_workers=0."""
    source = DummySource.Config()
    source.num_samples = 5
    config = DataPipeline.Config(source=source)
    pipeline = DataPipeline(config)

    loader = pipeline.create_loader(num_workers=0)

    assert loader is not None
    assert loader.num_workers == 0


def test_create_loader_with_workers():
    """Test create_loader with num_workers>0."""
    source = DummySource.Config()
    source.num_samples = 10
    config = DataPipeline.Config(source=source)
    pipeline = DataPipeline(config)

    loader = pipeline.create_loader(num_workers=2, prefetch_factor=1)

    assert loader is not None
    assert loader.num_workers == 2


def test_create_loader_with_cache():
    """Test create_loader with caching enabled."""
    source = DummySource.Config()
    source.num_samples = 5
    config = DataPipeline.Config(source=source)
    config.enable_cache = True
    pipeline = DataPipeline(config)

    loader = pipeline.create_loader(num_workers=0)

    assert loader is not None
    # Should return cached list loader.
    assert loader.num_workers == 0


def test_worker_dataset_single_worker():
    """Test _SingleWorkerDataset with single worker (no worker info)."""
    source = DummySource.Config()
    source.num_samples = 5
    config = DataPipeline.Config(source=source)
    pipeline = DataPipeline(config)

    dataset = _SingleWorkerDataset(pipeline)
    samples = list(dataset)

    # Should get all 5 samples.
    assert len(samples) == 5


# DataLoader spawn-based smoke tests: each real subprocess worker adds
# ~3-7s on macOS (spawn start method) and they account for ~half the
# total pytest wall time. The logic they cover (__iter__ under simulated
# worker info) is already exercised by the faster mock-based tests
# below (test_worker_dataset_multiworker_with_mock, ...). Mark these
# as `integration` so they run in CI but not in the default inner loop.


@pytest.mark.cli_python_subprocess
def test_worker_dataset_with_worker_info():
    """Test _MultipleWorkerDataset with simulated worker info."""
    source = DummySource.Config()
    source.num_samples = 10
    config = DataPipeline.Config(source=source)

    # Create a DataLoader with workers to test multi-worker scenario.
    loader = torch.utils.data.DataLoader(
        _MultipleWorkerDataset(config),
        num_workers=1,
        multiprocessing_context=_WORKER_CONTEXT,
    )

    # Collect all samples from loader.
    samples = list(loader)

    # Should get all samples distributed across workers.
    assert len(samples) > 0


@pytest.mark.cli_python_subprocess
def test_worker_dataset_with_sliceable_source():
    """Test _MultipleWorkerDataset with source that supports slicing."""
    # Test with data_dir that has parquet files.
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # Create fewer parquet files than workers to trigger warning.
        (data_dir / "shard_0.parquet").touch()
        (data_dir / "shard_1.parquet").touch()

        source = SliceableSource.Config()
        source.num_samples = 20
        source.data_dir = str(data_dir)
        config = DataPipeline.Config(source=source)

        loader = torch.utils.data.DataLoader(
            _MultipleWorkerDataset(config),
            num_workers=1,
            multiprocessing_context=_WORKER_CONTEXT,
        )

        samples = list(loader)
        assert len(samples) > 0


@pytest.mark.cli_python_subprocess
def test_worker_dataset_non_sliceable_source_single_worker_ok():
    """A non-sliceable source is fine with a single worker (num_workers=1)."""
    # num_workers=1 -> num_workers_inner==1, no slicing required.
    source = DummySource.Config()
    source.num_samples = 10
    config = DataPipeline.Config(source=source)

    loader = torch.utils.data.DataLoader(
        _MultipleWorkerDataset(config),
        num_workers=1,
        multiprocessing_context=_WORKER_CONTEXT,
    )

    samples = list(loader)
    assert len(samples) > 0


@pytest.mark.cli_python_subprocess
def test_worker_dataset_with_nonexistent_data_dir():
    """Test _MultipleWorkerDataset when data_dir doesn't exist."""
    source = SliceableSourceWithBadPath.Config()
    config = DataPipeline.Config(source=source)

    loader = torch.utils.data.DataLoader(
        _MultipleWorkerDataset(config),
        num_workers=1,
        multiprocessing_context=_WORKER_CONTEXT,
    )

    samples = list(loader)
    assert len(samples) > 0


def test_add_filter_reason():
    """Test add_filter_reason function (lines 75-77)."""
    sample: dict[str, object] = {"key": "test"}

    # First call - should create filter_reasons list.
    add_filter_reason(sample, "filter1", "reason1")
    assert "filter_reasons" in sample
    assert sample["filter_reasons"] == ["filter1:reason1"]

    # Second call - should append.
    add_filter_reason(sample, "filter2", "reason2")
    assert sample["filter_reasons"] == ["filter1:reason1", "filter2:reason2"]


def test_add_filter_reason_typed():
    """Test add_filter_reason_typed function (line 84)."""
    sample: Sample = {"key": "test"}  # TypedDict.

    # This should work with TypedDict.
    add_filter_reason_typed(sample, "filter1", "reason1")
    assert "filter_reasons" in sample
    assert sample["filter_reasons"] == ["filter1:reason1"]


def test_add_filter_reason_skips_after_decode_failed():
    """Test that add_filter_reason skips adding reasons after decode_failed."""
    sample: dict[str, object] = {"key": "test"}

    add_filter_reason(sample, "CropDuringDecodeImage", "decode_failed")
    assert sample["filter_reasons"] == ["CropDuringDecodeImage:decode_failed"]

    add_filter_reason(sample, "CLIPEmbedding", "missing_media_tensor")
    add_filter_reason(sample, "SigLIPEmbedding", "missing_media_tensor")
    add_filter_reason(sample, "DINOEmbedding", "missing_media_tensor")

    assert sample["filter_reasons"] == ["CropDuringDecodeImage:decode_failed"]


def test_worker_dataset_multiworker_with_mock():
    """Test _MultipleWorkerDataset multi-worker paths with mocking to get coverage."""
    # Test with mocked worker info to get coverage.
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # Create fewer parquet files than workers.
        (data_dir / "shard_0.parquet").touch()
        (data_dir / "shard_1.parquet").touch()

        source = SliceableSource.Config()
        source.num_samples = 20
        source.data_dir = str(data_dir)
        config = DataPipeline.Config(source=source)

        # Mock worker info to simulate multi-worker environment.
        mock_worker_info = Mock()
        mock_worker_info.id = 0
        mock_worker_info.num_workers = 4  # More workers than parquet files.

        dataset = _MultipleWorkerDataset(config)

        with patch("torch.utils.data.get_worker_info", return_value=mock_worker_info):
            samples = list(dataset)
            # Should get samples for worker 0.
            assert len(samples) > 0


def test_worker_dataset_non_sliceable_multiworker_raises():
    """Non-sliceable source with num_workers>1 raises instead of duplicating (H12)."""
    source = DummySource.Config()
    source.num_samples = 10
    config = DataPipeline.Config(source=source)

    # Mock worker info to simulate multi-worker environment.
    mock_worker_info = Mock()
    mock_worker_info.id = 0
    mock_worker_info.num_workers = 2

    dataset = _MultipleWorkerDataset(config)

    with (
        patch("torch.utils.data.get_worker_info", return_value=mock_worker_info),
        pytest.raises(TypeError, match="does not support slicing"),
    ):
        list(dataset)


def test_datapipeline_len_raises_for_non_sized_source():
    """__len__ raises TypeError for a source without __len__ (H14)."""

    class _Unsized:
        class Config(Fig["_Unsized"]): ...

        def __init__(self, config: _Unsized.Config) -> None:
            del config

        def __iter__(self) -> Iterator[Sample]:
            yield cast(Sample, {"key": "a"})

    config = DataPipeline.Config(source=_Unsized.Config())
    pipeline = DataPipeline(config)

    with pytest.raises(TypeError, match="not Sized"):
        len(pipeline)


def test_datapipeline_len_uses_sized_source():
    """__len__ returns the source length when the source is Sized."""
    source = DummySource.Config()
    source.num_samples = 7
    pipeline = DataPipeline(DataPipeline.Config(source=source))
    assert len(pipeline) == 7


def _mock_dp_mesh(rank: int, world: int) -> Mock:
    """Mock a device mesh whose ``dp`` dimension reports ``(rank, world)``."""
    dp = Mock()
    dp.get_local_rank.return_value = rank
    dp.size.return_value = world
    mesh = Mock()
    mesh.mesh_dim_names = ("dp",)
    mesh.__getitem__ = Mock(return_value=dp)
    return mesh


def _emit_indices(
    num_samples: int,
    dp_rank: int,
    dp_world: int,
    worker_id: int,
    num_workers: int,
) -> list[int]:
    """Sample indices a single (dp_rank, worker) shard emits from the pipeline."""
    source = SliceableSource.Config()
    source.num_samples = num_samples
    dataset = _MultipleWorkerDataset(DataPipeline.Config(source=source))

    worker_info = Mock()
    worker_info.id = worker_id
    worker_info.num_workers = num_workers

    with (
        patch(
            "priml.data.pipeline.dataset.global_device_mesh",
            return_value=_mock_dp_mesh(dp_rank, dp_world),
        ),
        patch("torch.utils.data.get_worker_info", return_value=worker_info),
    ):
        return [int(str(s["key"]).removeprefix("sample_")) for s in dataset]


def test_distributed_dp_rank_partitions_data_exactly_once() -> None:
    """#317: composing dp rank with worker shard partitions the dataset once.

    Without the dp factor every replica slices only by its local worker index,
    so all ranks emit the SAME items (replication). The union across every
    (dp_rank, worker) shard must equal the full dataset with no gap/overlap.
    """
    num_samples, dp_world, num_workers = 60, 3, 2
    union: list[int] = []
    per_rank: dict[int, set[int]] = {}
    for dp_rank in range(dp_world):
        rank_items: set[int] = set()
        for worker_id in range(num_workers):
            idx = _emit_indices(num_samples, dp_rank, dp_world, worker_id, num_workers)
            union.extend(idx)
            rank_items.update(idx)
        per_rank[dp_rank] = rank_items

    # Exact partition: every item once, no duplicates.
    assert sorted(union) == list(range(num_samples))
    assert len(union) == num_samples
    # Distinct dp ranks get DISJOINT items (the bug had them identical).
    assert per_rank[0].isdisjoint(per_rank[1])
    assert per_rank[0].isdisjoint(per_rank[2])
    assert per_rank[1].isdisjoint(per_rank[2])


def test_distributed_dp_rank_shards_with_num_workers_zero() -> None:
    """#317: dp sharding applies even with num_workers=0 (no DataLoader workers).

    DDP with a non-worker DataLoader still requires per-rank disjoint data.
    """
    num_samples, dp_world = 30, 3
    union: list[int] = []
    for dp_rank in range(dp_world):
        # worker_info is None -> worker_id=0, num_workers_inner=1.
        source = SliceableSource.Config()
        source.num_samples = num_samples
        dataset = _MultipleWorkerDataset(DataPipeline.Config(source=source))
        with (
            patch(
                "priml.data.pipeline.dataset.global_device_mesh",
                return_value=_mock_dp_mesh(dp_rank, dp_world),
            ),
            patch("torch.utils.data.get_worker_info", return_value=None),
        ):
            union.extend(int(str(s["key"]).removeprefix("sample_")) for s in dataset)
    assert sorted(union) == list(range(num_samples))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
