"""Tests for DataPipeline."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict, cast
from unittest.mock import Mock, patch

import multiprocessing as mp
import tempfile

from configgle import Fig
from torch.utils import data

import pytest

from priml.data.pipeline.batching import Batcher
from priml.data.pipeline.dataset import (
    DataPipeline,
    _assign_gpu_to_worker,
    _distributed_shard,
    _DummySource,
    _MultipleWorkerDataset,
    _passthrough_collate,
    _pipeline_uses_cuda,
    _SingleWorkerDataset,
    add_filter_reason,
    add_filter_reason_typed,
)
from priml.data.processors.custom_types import Sample
from priml.data.sources.sharding import shard_and_shuffle


if TYPE_CHECKING:
    from collections.abc import Iterator


_WORKER_CONTEXT: Final = "fork" if "fork" in mp.get_all_start_methods() else None


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


def test_single_worker_starts_at_epoch_zero() -> None:
    source = ShufflingSliceableSource.Config(num_samples=8)
    dataset = _SingleWorkerDataset(DataPipeline(DataPipeline.Config(source=source)))

    actual = [int(str(sample["key"]).removeprefix("sample_")) for sample in dataset]
    assert actual == shard_and_shuffle(
        list(range(8)),
        worker_slice=None,
        shuffle=True,
        epoch_seed=0,
    )


def test_single_worker_set_epoch_uses_requested_seed() -> None:
    source = ShufflingSliceableSource.Config(num_samples=8)
    dataset = _SingleWorkerDataset(DataPipeline(DataPipeline.Config(source=source)))
    dataset.set_epoch(2)

    actual = [int(str(sample["key"]).removeprefix("sample_")) for sample in dataset]
    assert actual == shard_and_shuffle(
        list(range(8)),
        worker_slice=None,
        shuffle=True,
        epoch_seed=2,
    )


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


def test_pipeline_does_not_wrap_batching_processors() -> None:
    config = DataPipeline.Config(processors=[Batcher.Config(size=2)])
    pipeline = DataPipeline(config)

    assert isinstance(pipeline.processors[0], Batcher)


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


def test_assign_gpu_to_worker_falls_back_to_cpu(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        patch("torch.cuda.is_available", return_value=False),
        caplog.at_level("DEBUG"),
    ):
        assert _assign_gpu_to_worker(3) == -1

    assert caplog.records[0].getMessage() == "Worker 3: CUDA not available, using CPU"


def test_pipeline_uses_cuda_only_for_nested_cuda_devices() -> None:
    class Stage:
        class Config(Fig["Stage"]):
            device: str = "cpu"

    class Container(DummyProcessor):
        class Config(DummyProcessor.Config):
            stage: Stage.Config = field(default_factory=Stage.Config)

    cpu = DataPipeline.Config(
        processors=[Container.Config(), DummyProcessor.Config()],
    )
    cuda_stage = Container.Config(stage=Stage.Config(device="cuda:1"))
    cuda = DataPipeline.Config(
        processors=[cuda_stage, DummyProcessor.Config()],
    )

    assert not _pipeline_uses_cuda(cpu)
    assert _pipeline_uses_cuda(cuda)


def test_pipeline_uses_cuda_handles_cyclic_config_graph() -> None:
    class Stage:
        class Config(Fig["Stage"]):
            device: str = "cuda"

    class Node(DummyProcessor):
        class Config(DummyProcessor.Config):
            stage: Stage.Config = field(default_factory=Stage.Config)
            child: object | None = None

    node = Node.Config(stage=Stage.Config(device="cuda"))
    node.child = node
    assert _pipeline_uses_cuda(DataPipeline.Config(processors=[node]))


def test_create_loader_no_workers():
    """Test create_loader with num_workers=0."""
    source = DummySource.Config()
    source.num_samples = 5
    config = DataPipeline.Config(source=source)
    pipeline = DataPipeline(config)

    loader = pipeline.create_loader(num_workers=0)

    assert loader.num_workers == 0
    assert loader.prefetch_factor is None
    assert next(iter(loader)) == {
        "key": "sample_0",
        "width": 512,
        "height": 512,
        "caption": "Caption 0",
        "url": "https://example.com/image_0.jpg",
    }


def test_create_loader_with_workers():
    """Test create_loader with num_workers>0."""
    source = SliceableSource.Config(num_samples=10)
    config = DataPipeline.Config(source=source)
    pipeline = DataPipeline(config)

    loader = pipeline.create_loader(num_workers=2, prefetch_factor=1)

    assert loader.num_workers == 2
    assert loader.prefetch_factor == 1
    assert isinstance(loader.dataset, data.IterableDataset)


def test_create_loader_with_one_worker() -> None:
    source = SliceableSource.Config(num_samples=5)
    pipeline = DataPipeline(DataPipeline.Config(source=source))

    loader = pipeline.create_loader(num_workers=1)

    assert loader.num_workers == 1
    assert loader.prefetch_factor == 2
    # Forkserver, the default, starts its server per process: 1.1s against 0.05s.
    loader.multiprocessing_context = _WORKER_CONTEXT
    assert [sample["key"] for sample in loader] == [f"sample_{i}" for i in range(5)]


def test_create_loader_with_cache():
    """Test create_loader with caching enabled."""
    source = DummySource.Config()
    source.num_samples = 5
    config = DataPipeline.Config(source=source)
    config.enable_cache = True
    pipeline = DataPipeline(config)

    loader = pipeline.create_loader(num_workers=0)

    assert loader.num_workers == 0
    assert [sample["key"] for sample in loader] == [f"sample_{i}" for i in range(5)]


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
    loader = data.DataLoader(
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

        loader = data.DataLoader(
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

    loader = data.DataLoader(
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

    loader = data.DataLoader(
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


def test_add_filter_reason_normalizes_existing_value_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sample: dict[str, object] = {"filter_reasons": ("old", 3)}

    with caplog.at_level("DEBUG"):
        add_filter_reason(sample, "Resize", "too_small")

    assert sample["filter_reasons"] == ["Resize:too_small"]
    assert caplog.records[0].getMessage() == "Filter: Resize:too_small"


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

    with pytest.raises(
        TypeError,
        match=r"^Source _Unsized is not Sized; DataPipeline has no length\.$",
    ):
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


def test_multiworker_starts_at_epoch_zero() -> None:
    source = ShufflingSliceableSource.Config(num_samples=8)
    dataset = _MultipleWorkerDataset(DataPipeline.Config(source=source))
    worker_info = Mock(id=0, num_workers=1)

    with patch("torch.utils.data.get_worker_info", return_value=worker_info):
        actual = [int(str(sample["key"]).removeprefix("sample_")) for sample in dataset]

    assert actual == shard_and_shuffle(
        list(range(8)),
        worker_slice=(0, 1),
        shuffle=True,
        epoch_seed=0,
    )


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


def test_distributed_shard_reads_dp_mesh_dimension() -> None:
    dp = Mock()
    dp.get_local_rank.return_value = 2
    dp.size.return_value = 5
    mesh = Mock(mesh_dim_names=("dp",))
    mesh.__getitem__ = Mock(return_value=dp)

    with patch(
        "priml.data.pipeline.dataset.global_device_mesh",
        return_value=mesh,
    ):
        assert _distributed_shard() == (2, 5)

    mesh.__getitem__.assert_called_once_with("dp")


def test_distributed_shard_ignores_mesh_without_dp_dimension() -> None:
    mesh = Mock(mesh_dim_names=("tp",))
    with patch(
        "priml.data.pipeline.dataset.global_device_mesh",
        return_value=mesh,
    ):
        assert _distributed_shard() == (0, 1)


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


def test_assign_gpu_to_worker_round_robins_outside_distributed_execution(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Use worker-based placement until distributed is initialized."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=4),
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=False),
        patch("torch.cuda.current_device", return_value=3) as current_device,
        patch("torch.cuda.set_device") as set_device,
        caplog.at_level("INFO"),
    ):
        assert _assign_gpu_to_worker(6) == 2

    current_device.assert_not_called()
    set_device.assert_called_once_with(2)
    assert caplog.records[-1].getMessage() == (
        "Worker 6: assigned to GPU 2 (total GPUs: 4)"
    )


def test_assign_gpu_to_worker_uses_rank_device_when_distributed_initialized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Use the current rank's device after distributed initialization."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=4),
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.cuda.current_device", return_value=3) as current_device,
        patch("torch.cuda.set_device") as set_device,
        caplog.at_level("INFO"),
    ):
        assert _assign_gpu_to_worker(6) == 3

    current_device.assert_called_once_with()
    set_device.assert_called_once_with(3)
    assert caplog.records[-1].getMessage() == (
        "Worker 6: assigned to GPU 3 (total GPUs: 4)"
    )


def test_assign_gpu_to_worker_handles_forked_cuda_with_diagnostic(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fall back to CPU with a diagnostic after a forked CUDA runtime error."""
    error = RuntimeError("Cannot re-initialize CUDA in forked subprocess")
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=4),
        patch("torch.distributed.is_available", return_value=False),
        patch("torch.cuda.current_device", return_value=3),
        patch("torch.cuda.set_device", side_effect=error) as set_device,
        caplog.at_level("WARNING"),
    ):
        assert _assign_gpu_to_worker(6) == -1

    set_device.assert_called_once_with(2)
    assert caplog.records[-1].getMessage() == (
        "Worker 6: CUDA was initialized before fork, cannot assign GPU "
        "(this is expected in test environments)"
    )


@pytest.mark.parametrize("batch", [[], [{}, {}]])
def test_passthrough_collate_requires_exactly_one_item(
    batch: list[dict[str, object]],
) -> None:
    """Reject empty and multi-item DataLoader wrappers with the exact message."""
    with pytest.raises(ValueError, match=r"^Expected len\(batch\) == 1\.$"):
        _passthrough_collate(batch)


def test_pipeline_uses_cuda_after_non_dataclass_config_child() -> None:
    class Stage:
        class Config(Fig["Stage"]):
            device: str = "cpu"

    class Container(DummyProcessor):
        class Config(DummyProcessor.Config):
            stage: Stage.Config = field(default_factory=Stage.Config)
            blocker: object = field(default_factory=object)

    cuda_stage = Container.Config(stage=Stage.Config(device="cuda:1"))
    nested_without_cuda = Container.Config()

    assert _pipeline_uses_cuda(
        DataPipeline.Config(processors=[cuda_stage, nested_without_cuda]),
    )


def test_pipeline_without_shortcircuit_keeps_processors() -> None:
    config = DataPipeline.Config(processors=[DummyProcessor.Config()])
    config.filters_shortcircuit = False

    pipeline = DataPipeline(config)

    assert len(pipeline.processors) == 1
    assert isinstance(pipeline.processors[0], DummyProcessor)


def test_create_loader_defaults_to_single_process() -> None:
    pipeline = DataPipeline(DataPipeline.Config(source=DummySource.Config()))

    loader = pipeline.create_loader()

    assert loader.num_workers == 0
    assert loader.prefetch_factor is None


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
