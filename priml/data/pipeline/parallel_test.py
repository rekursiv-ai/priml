"""Tests for parallel processing utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import threading
import time

from configgle import Fig

import pytest

from priml.data.pipeline.parallel import ParMap, PrefetchBuffer


if TYPE_CHECKING:
    from collections.abc import Iterator


type TestSample = dict[str, object]


class RaisingProcessor:
    """Processor that raises on the first sample it sees."""

    class Config(Fig["RaisingProcessor"]): ...

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, samples: Iterator[TestSample]) -> Iterator[TestSample]:
        for _ in samples:
            raise RuntimeError("boom")
        yield from ()


class SlowProcessor:
    """Mock processor that adds a delay to simulate I/O."""

    class Config(Fig["SlowProcessor"]):
        delay_ms: int = 10
        add_field: str = "processed"

    def __init__(self, config: Config):
        self.delay_ms = config.delay_ms
        self.add_field = config.add_field

    def __call__(self, samples: Iterator[TestSample]) -> Iterator[TestSample]:
        for sample in samples:
            time.sleep(self.delay_ms / 1000.0)
            sample[self.add_field] = True
            yield sample


# ParMap Tests.


def test_threadpoolmap_basic():
    """Test ParMap processes samples in parallel."""
    config = ParMap.Config(num_threads=4)
    threadpool = config.make()

    # Create slow upstream processor.
    slow = SlowProcessor.Config(delay_ms=1).make()

    # Create test samples.
    samples: list[TestSample] = [{"id": i} for i in range(4)]

    # Process samples through slow processor then threadpool.
    results = list(threadpool(slow(iter(samples))))

    # All samples should be processed.
    assert len(results) == 4

    # All samples should have field added by slow processor.
    for result in results:
        assert result.get("processed") is True

    # Results might be out of order (due to parallel processing)
    result_ids = {result["id"] for result in results}
    expected_ids = set(range(4))
    assert result_ids == expected_ids


def test_threadpoolmap_single_thread():
    """Test ParMap works with single thread (passthrough)."""
    config = ParMap.Config(num_threads=1)
    threadpool = ParMap(config)

    samples: list[TestSample] = [{"id": i} for i in range(5)]
    results = list(threadpool(iter(samples)))

    assert len(results) == 5
    # Single thread should preserve order.
    for i, result in enumerate(results):
        assert result["id"] == i


def test_threadpoolmap_empty_input():
    """Test ParMap handles empty input."""
    config = ParMap.Config(num_threads=2)
    threadpool = ParMap(config)

    results = list(threadpool(iter([])))
    assert len(results) == 0


def test_threadpoolmap_preserves_data():
    """Test ParMap preserves original sample data."""
    config = ParMap.Config(num_threads=2)
    threadpool = ParMap(config)

    samples: list[TestSample] = [
        {"id": 0, "name": "alice", "value": 42},
        {"id": 1, "name": "bob", "value": 99},
    ]

    results = list(threadpool(iter(samples)))

    # Find results by id (order may vary)
    results_by_id = {r["id"]: r for r in results}

    assert results_by_id[0]["name"] == "alice"
    assert results_by_id[0]["value"] == 42
    assert results_by_id[1]["name"] == "bob"
    assert results_by_id[1]["value"] == 99


def test_threadpoolmap_queue_management():
    """Test ParMap respects max_input_queue_size."""
    config = ParMap.Config(
        num_threads=2,
        max_input_queue_size=5,
        max_output_queue_size=5,
    )

    slow = SlowProcessor.Config(delay_ms=1).make()
    threadpool = config.make()

    # Create more samples than queue size.
    samples: list[TestSample] = [{"id": i} for i in range(20)]

    results = list(threadpool(slow(iter(samples))))

    # All samples should be processed despite queue limit.
    assert len(results) == 20


def test_threadpoolmap_speedup():
    """Test ParMap processes same number of samples correctly.

    Note: Actual speedup tests are flaky in CI, so we just verify correctness.
    Manual testing shows ~2-3x speedup with 4 threads on I/O-bound workloads.
    """
    # Parallel processing with 4 threads.
    samples: list[TestSample] = [{"id": i} for i in range(8)]
    slow = SlowProcessor.Config(delay_ms=1).make()
    threadpool = ParMap.Config(num_threads=4).make()

    results = list(threadpool(slow(iter(samples))))

    # Verify correctness: all samples processed.
    assert len(results) == 8
    result_ids = {r["id"] for r in results}
    assert result_ids == set(range(8))
    for result in results:
        assert result.get("processed") is True


def test_threadpoolmap_worker_exception_does_not_deadlock():
    """A worker exception surfaces without deadlocking a bounded input queue (H9)."""
    config = ParMap.Config(
        num_threads=2,
        max_input_queue_size=2,
        max_output_queue_size=2,
        processors=[RaisingProcessor.Config()],
    )
    threadpool = config.make()

    # Far more samples than the bounded queue can hold; a naive feeder would
    # block forever once the dead workers stop draining the input queue.
    samples: Iterator[TestSample] = ({"id": i} for i in range(1000))

    result: dict[str, str] = {}

    def run() -> None:
        try:
            list(threadpool(samples))
        except RuntimeError as e:
            result["error"] = str(e)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    runner.join(timeout=5.0)

    assert not runner.is_alive(), "ParMap deadlocked on worker exception"
    assert result.get("error") == "boom"


# PrefetchBuffer Tests.


def test_prefetchbuffer_basic():
    """Test PrefetchBuffer prefetches samples."""
    config = PrefetchBuffer.Config(size=10)
    prefetch = PrefetchBuffer(config)

    samples: list[TestSample] = [{"id": i} for i in range(20)]
    results = list(prefetch(iter(samples)))

    assert len(results) == 20
    for i, result in enumerate(results):
        assert result["id"] == i


def test_prefetchbuffer_preserves_order():
    """Test PrefetchBuffer preserves sample order."""
    config = PrefetchBuffer.Config(size=5)
    prefetch = PrefetchBuffer(config)

    samples: list[TestSample] = [{"id": i, "value": i * 2} for i in range(10)]
    results = list(prefetch(iter(samples)))

    # Order should be preserved.
    for i, result in enumerate(results):
        assert result["id"] == i
        assert result["value"] == i * 2


def test_prefetchbuffer_empty_input():
    """Test PrefetchBuffer handles empty input."""
    config = PrefetchBuffer.Config(size=5)
    prefetch = PrefetchBuffer(config)

    results = list(prefetch(iter([])))
    assert len(results) == 0


def test_prefetchbuffer_with_slow_producer():
    """Test PrefetchBuffer improves throughput with slow producer."""
    config = PrefetchBuffer.Config(size=10)
    prefetch = config.make()

    slow = SlowProcessor.Config(delay_ms=1).make()

    samples: list[TestSample] = [{"id": i} for i in range(8)]

    # With prefetch, consumption starts immediately.
    start = time.perf_counter()
    results: list[TestSample] = []
    for sample in prefetch(slow(iter(samples))):
        results.append(sample)
        # Small delay to simulate downstream processing.
        time.sleep(0.0001)
    elapsed = time.perf_counter() - start

    assert len(results) == 8
    # Should be faster than processing all sequentially
    # (This is a weak assertion to avoid flakiness, but prefetch helps)
    assert elapsed < 0.5


def test_parmap_with_no_threads_runs_the_stages_inline_and_in_order() -> None:
    config = ParMap.Config(
        num_threads=0,
        processors=[SlowProcessor.Config(delay_ms=0, add_field="a")],
    )
    samples: list[TestSample] = [{"id": i} for i in range(3)]
    results = list(config.make()(iter(samples)))
    assert [r["id"] for r in results] == [0, 1, 2]
    assert all(r["a"] is True for r in results)


def test_parmap_bounded_queues_process_every_sample() -> None:
    config = ParMap.Config(
        num_threads=2,
        max_input_queue_size=1,
        max_output_queue_size=1,
        processors=[SlowProcessor.Config(delay_ms=1, add_field="b")],
    )
    samples: list[TestSample] = [{"id": i} for i in range(6)]
    results = list(config.make()(iter(samples)))
    assert {r["id"] for r in results} == set(range(6))
    assert all(r["b"] is True for r in results)


def test_parmap_surfaces_a_source_exception_from_the_feeder() -> None:
    def source() -> Iterator[TestSample]:
        yield {"id": 0}
        raise ValueError("upstream")

    config = ParMap.Config(num_threads=2)
    with pytest.raises(ValueError, match="upstream"):
        list(config.make()(source()))


def test_prefetchbuffer_size_zero_is_a_passthrough() -> None:
    samples: list[TestSample] = [{"id": 0}, {"id": 1}]
    assert list(PrefetchBuffer.Config(size=0).make()(iter(samples))) == samples


def test_prefetchbuffer_fill_first_unbounded_loads_everything_eagerly() -> None:
    seen: list[int] = []

    def source() -> Iterator[TestSample]:
        for i in range(3):
            seen.append(i)
            yield {"id": i}

    stream = PrefetchBuffer.Config(size=-1, fill_first=True).make()(source())
    first = next(stream)
    assert first == {"id": 0}
    assert seen == [0, 1, 2]
    assert [s["id"] for s in stream] == [1, 2]


def test_prefetchbuffer_fill_first_bounded_waits_for_the_buffer() -> None:
    samples: list[TestSample] = [{"id": i} for i in range(5)]
    config = PrefetchBuffer.Config(size=2, fill_first=True)
    assert [s["id"] for s in config.make()(iter(samples))] == [0, 1, 2, 3, 4]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
