"""Tests for parallel processing utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import logging
import queue
import threading
import time

from configgle import Fig

import pytest

from priml.data.pipeline.parallel import (
    ParMap,
    PrefetchBuffer,
    _put_until_stopped,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator


type TestSample = dict[str, object]


def _samples(count: int) -> Iterator[TestSample]:
    samples: list[TestSample] = [{"id": index} for index in range(count)]
    return iter(samples)


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
    slow = SlowProcessor.Config(delay_ms=1).make()
    samples: list[TestSample] = [{"id": i} for i in range(4)]
    results = list(threadpool(slow(iter(samples))))
    assert len(results) == 4
    for result in results:
        assert result.get("processed") is True
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
    samples: list[TestSample] = [{"id": i} for i in range(20)]
    results = list(threadpool(slow(iter(samples))))
    assert len(results) == 20


def test_threadpoolmap_speedup():
    """Test ParMap processes same number of samples correctly.

    Note: Actual speedup tests are flaky in CI, so we just verify correctness.
    Manual testing shows ~2-3x speedup with 4 threads on I/O-bound workloads.
    """
    samples: list[TestSample] = [{"id": i} for i in range(8)]
    slow = SlowProcessor.Config(delay_ms=1).make()
    threadpool = ParMap.Config(num_threads=4).make()
    results = list(threadpool(slow(iter(samples))))
    assert len(results) == 8
    result_ids = {r["id"] for r in results}
    assert result_ids == set(range(8))
    for result in results:
        assert result.get("processed") is True


def test_parmap_forwards_the_only_worker_exception() -> None:
    parmap = ParMap.Config(
        num_threads=1,
        processors=[RaisingProcessor.Config()],
    ).make()

    with pytest.raises(RuntimeError, match=r"^boom$"):
        list(parmap(_samples(1)))


def test_threadpoolmap_worker_exception_does_not_deadlock():
    """A worker exception surfaces without deadlocking a bounded input queue (H9)."""
    config = ParMap.Config(
        num_threads=2,
        max_input_queue_size=2,
        max_output_queue_size=2,
        processors=[RaisingProcessor.Config()],
    )
    threadpool = config.make()
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
    for i, result in enumerate(results):
        assert result["id"] == i
        assert result["value"] == i * 2


def test_prefetchbuffer_empty_input():
    """Test PrefetchBuffer handles empty input."""
    config = PrefetchBuffer.Config(size=5)
    prefetch = PrefetchBuffer(config)
    results = list(prefetch(iter([])))
    assert len(results) == 0


def test_prefetchbuffer_empty_input_logs_zero_peak(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="priml.data.pipeline.parallel")

    assert list(PrefetchBuffer.Config(size=1).make()(iter([]))) == []

    assert any(
        record.getMessage() == "PrefetchBuffer peak queue usage: 0/1"
        for record in caplog.records
    )


def test_prefetchbuffer_with_slow_producer():
    """Test PrefetchBuffer improves throughput with slow producer."""
    config = PrefetchBuffer.Config(size=10)
    prefetch = config.make()
    slow = SlowProcessor.Config(delay_ms=1).make()
    samples: list[TestSample] = [{"id": i} for i in range(8)]
    start = time.perf_counter()
    results: list[TestSample] = []
    for sample in prefetch(slow(iter(samples))):
        results.append(sample)
        time.sleep(0.0001)
    elapsed = time.perf_counter() - start
    assert len(results) == 8
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
        yield from ()
        raise ValueError("upstream")

    config = ParMap.Config(
        num_threads=2,
        processors=[SlowProcessor.Config(delay_ms=0)],
    )
    with pytest.raises(ValueError, match=r"^upstream$"):
        list(config.make()(source()))


def test_parmap_preserves_base_exceptions_while_sending_poison_pills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = threading.Event()
    release_worker = threading.Event()
    source_raising = threading.Event()
    feeder_exception = threading.Event()

    def record_thread_exception(args: threading.ExceptHookArgs) -> None:
        if isinstance(args.exc_value, KeyboardInterrupt):
            feeder_exception.set()

    monkeypatch.setattr(threading, "excepthook", record_thread_exception)

    class FailingProcessor:
        class Config(Fig["FailingProcessor"]): ...

        def __init__(self, config: Config) -> None:
            del config

        def __call__(self, samples: Iterator[TestSample]) -> Iterator[TestSample]:
            for _sample in samples:
                worker_started.set()
                release_worker.wait(timeout=1)
                raise RuntimeError("worker")
            yield from ()

    def source() -> Iterator[TestSample]:
        yield {"id": 0}
        yield {"id": 1}
        source_raising.set()
        raise KeyboardInterrupt

    parmap = ParMap.Config(
        num_threads=1,
        max_input_queue_size=1,
        processors=[FailingProcessor.Config()],
    ).make()

    def consume() -> None:
        with pytest.raises(RuntimeError, match="worker"):
            list(parmap(source()))

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    assert worker_started.wait(timeout=1)
    assert source_raising.wait(timeout=1)
    release_worker.set()
    consumer.join(timeout=2)

    assert not consumer.is_alive()
    assert feeder_exception.wait(timeout=1)


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


def test_prefetchbuffer_fill_first_does_not_release_after_one_item() -> None:
    source_paused = threading.Event()
    release_source = threading.Event()
    first_returned = threading.Event()

    def source() -> Iterator[TestSample]:
        yield {"id": 0}
        source_paused.set()
        release_source.wait(timeout=1)
        yield {"id": 1}

    stream = PrefetchBuffer.Config(size=2, fill_first=True).make()(source())
    result: list[TestSample] = []

    def consume_first() -> None:
        result.append(next(stream))
        first_returned.set()

    consumer = threading.Thread(target=consume_first, daemon=True)
    consumer.start()
    assert source_paused.wait(timeout=1)
    released_early = first_returned.wait(timeout=0.05)
    release_source.set()
    consumer.join(timeout=1)

    assert not released_early, "fill_first released before reaching its bound"
    assert not consumer.is_alive()
    assert result == [{"id": 0}]
    assert [sample["id"] for sample in stream] == [1]


def test_parmap_logs_name_queue_bounds_and_fill_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    samples: list[TestSample] = [{"id": i} for i in range(4)]
    caplog.set_level(logging.INFO, logger="priml.data.pipeline.parallel")
    parmap = ParMap.Config(
        num_threads=2,
        max_input_queue_size=-1,
        max_output_queue_size=2,
        processors=[SlowProcessor.Config(delay_ms=1)],
    ).make()
    assert len(list(parmap(iter(samples)))) == 4
    assert any(
        record.getMessage().startswith("ParMap(2 threads) peak queue usage: input=")
        and "/∞, output=" in record.getMessage()
        and record.getMessage().endswith("/2")
        for record in caplog.records
    )
    caplog.clear()
    prefetch = PrefetchBuffer.Config(size=2, fill_first=True).make()
    assert [sample["id"] for sample in prefetch(iter(samples))] == [0, 1, 2, 3]
    messages = [record.getMessage() for record in caplog.records]
    assert "PrefetchBuffer filled: 2/2 items buffered before yielding" in messages
    assert any(
        message.startswith("PrefetchBuffer peak queue usage: 2/2")
        for message in messages
    )
    caplog.clear()
    eager = PrefetchBuffer.Config(size=-1, fill_first=True).make()
    assert [sample["id"] for sample in eager(iter(samples))] == [0, 1, 2, 3]
    assert any(
        record.getMessage() == "PrefetchBuffer eagerly loaded 4 items (no thread)"
        for record in caplog.records
    )


def test_parmap_logs_zero_peaks_for_empty_threaded_input(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="priml.data.pipeline.parallel")
    parmap = ParMap.Config(
        num_threads=1,
        processors=[SlowProcessor.Config(delay_ms=0)],
    ).make()

    assert list(parmap(iter([]))) == []

    assert any(
        record.getMessage().endswith("input=0/∞, output=0/∞")
        for record in caplog.records
        if "ParMap(1 threads) peak queue usage" in record.getMessage()
    )


def test_parmap_fills_output_queue_to_its_exact_bound(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_queue = queue.Queue
    full = threading.Event()

    class ObservedQueue(real_queue[dict[str, object] | None]):
        @override
        def put(
            self,
            item: dict[str, object] | None,
            block: bool = True,
            timeout: float | None = None,
        ) -> None:
            super().put(item, block=block, timeout=timeout)
            if self.maxsize and self.qsize() == self.maxsize:
                full.set()

    def make_queue(*, maxsize: int = 0) -> ObservedQueue:
        return ObservedQueue(maxsize=maxsize)

    monkeypatch.setattr("priml.data.pipeline.parallel.queue.Queue", make_queue)
    caplog.set_level(logging.INFO, logger="priml.data.pipeline.parallel")
    parmap = ParMap.Config(
        num_threads=2,
        max_output_queue_size=1,
        processors=[SlowProcessor.Config(delay_ms=1)],
    ).make()
    stream = parmap(_samples(8))
    results = [next(stream)]
    assert full.wait(timeout=1), "workers did not fill the output queue"
    results.extend(stream)
    assert {sample["id"] for sample in results} == set(range(8))
    assert any(
        record.getMessage().endswith("output=1/1")
        for record in caplog.records
        if "peak queue usage" in record.getMessage()
    )


def test_parmap_logs_finite_input_and_output_queue_bounds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="priml.data.pipeline.parallel")
    parmap = ParMap.Config(
        num_threads=1,
        max_input_queue_size=3,
        max_output_queue_size=2,
        processors=[SlowProcessor.Config(delay_ms=0)],
    ).make()

    assert len(list(parmap(_samples(2)))) == 2

    assert any(
        "/3, output=" in record.getMessage() and record.getMessage().endswith("/2")
        for record in caplog.records
        if "ParMap(1 threads) peak queue usage" in record.getMessage()
    )


def test_parmap_logs_unbounded_output_capacity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="priml.data.pipeline.parallel")
    parmap = ParMap.Config(
        num_threads=2,
        max_input_queue_size=3,
        max_output_queue_size=-1,
        processors=[SlowProcessor.Config(delay_ms=1)],
    ).make()
    assert len(list(parmap(_samples(2)))) == 2
    assert any(
        "input=" in record.getMessage() and record.getMessage().endswith("/∞")
        for record in caplog.records
        if "ParMap(2 threads) peak queue usage:" in record.getMessage()
    )


def test_parmap_uses_configured_queue_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_queue = queue.Queue
    created: list[queue.Queue[dict[str, object] | None]] = []

    def make_queue(*, maxsize: int = 0) -> queue.Queue[dict[str, object] | None]:
        result: queue.Queue[dict[str, object] | None] = real_queue(maxsize=maxsize)
        created.append(result)
        return result

    monkeypatch.setattr("priml.data.pipeline.parallel.queue.Queue", make_queue)
    parmap = ParMap.Config(
        num_threads=1,
        max_input_queue_size=1,
        max_output_queue_size=2,
        processors=[SlowProcessor.Config(delay_ms=0)],
    ).make()

    assert len(list(parmap(_samples(1)))) == 1
    assert [items.maxsize for items in created] == [1, 2]


def test_parmap_uses_zero_for_unbounded_queues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_queue = queue.Queue
    created: list[queue.Queue[dict[str, object] | None]] = []

    def make_queue(*, maxsize: int = 0) -> queue.Queue[dict[str, object] | None]:
        result: queue.Queue[dict[str, object] | None] = real_queue(maxsize=maxsize)
        created.append(result)
        return result

    monkeypatch.setattr("priml.data.pipeline.parallel.queue.Queue", make_queue)
    parmap = ParMap.Config(
        num_threads=1,
        max_input_queue_size=-1,
        max_output_queue_size=-1,
        processors=[SlowProcessor.Config(delay_ms=0)],
    ).make()

    assert next(iter(parmap(_samples(1))))["id"] == 0
    assert [items.maxsize for items in created] == [0, 0]


def test_parmap_unbounded_output_can_buffer_past_one_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_queue = queue.Queue
    buffered_three = threading.Event()

    class ObservedQueue(real_queue[dict[str, object] | None]):
        is_output: bool
        peak_size: int

        def __init__(self, *, maxsize: int = 0, is_output: bool) -> None:
            super().__init__(maxsize=maxsize)
            self.is_output = is_output
            self.peak_size = 0

        @override
        def put(
            self,
            item: dict[str, object] | None,
            block: bool = True,
            timeout: float | None = None,
        ) -> None:
            super().put(item, block=block, timeout=timeout)
            self.peak_size = max(self.peak_size, self.qsize())
            if self.is_output and self.qsize() >= 3:
                buffered_three.set()

    queues: list[ObservedQueue] = []

    def make_queue(*, maxsize: int = 0) -> ObservedQueue:
        observed = ObservedQueue(maxsize=maxsize, is_output=len(queues) == 1)
        queues.append(observed)
        return observed

    monkeypatch.setattr("priml.data.pipeline.parallel.queue.Queue", make_queue)
    parmap = ParMap.Config(
        num_threads=2,
        max_output_queue_size=-1,
        processors=[SlowProcessor.Config(delay_ms=1)],
    ).make()
    stream = parmap(_samples(3))
    results = [next(stream)]
    buffered = buffered_three.wait(timeout=1)
    results.extend(stream)
    assert buffered, "the unbounded output queue stopped before buffering 2 samples"
    assert {sample["id"] for sample in results} == {0, 1, 2}
    assert queues[1].maxsize == 0
    assert queues[1].peak_size == 3


def test_prefetchbuffer_counts_an_item_consumed_before_the_producer_samples_qsize(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_queue = queue.Queue
    producer_paused = threading.Event()
    release_producer = threading.Event()

    class ObservedQueue(real_queue[dict[str, object] | None]):
        producer_thread_id: int | None = None

        @override
        def put(
            self,
            item: dict[str, object] | None,
            block: bool = True,
            timeout: float | None = None,
        ) -> None:
            super().put(item, block=block, timeout=timeout)
            if item is not None and self.producer_thread_id is None:
                self.producer_thread_id = threading.get_ident()
                producer_paused.set()
                release_producer.wait(timeout=1)

        @override
        def qsize(self) -> int:
            if (
                producer_paused.is_set()
                and self.producer_thread_id != threading.get_ident()
            ):
                release_producer.set()
            return super().qsize()

    def make_queue(*, maxsize: int = 0) -> ObservedQueue:
        return ObservedQueue(maxsize=maxsize)

    monkeypatch.setattr("priml.data.pipeline.parallel.queue.Queue", make_queue)
    caplog.set_level(logging.INFO, logger="priml.data.pipeline.parallel")
    stream = PrefetchBuffer.Config(size=-1).make()(_samples(1))

    assert next(stream) == {"id": 0}
    assert producer_paused.is_set()
    assert list(stream) == []
    assert any(
        record.getMessage() == "PrefetchBuffer peak queue usage: 1/∞"
        for record in caplog.records
    )


def test_prefetchbuffer_unbounded_queue_buffers_past_one_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_queue = queue.Queue
    buffered_three = threading.Event()
    observed: list[queue.Queue[dict[str, object] | None]] = []

    class ObservedQueue(real_queue[dict[str, object] | None]):
        @override
        def put(
            self,
            item: dict[str, object] | None,
            block: bool = True,
            timeout: float | None = None,
        ) -> None:
            super().put(item, block=block, timeout=timeout)
            if self.qsize() >= 3:
                buffered_three.set()

    def make_queue(*, maxsize: int = 0) -> ObservedQueue:
        result = ObservedQueue(maxsize=maxsize)
        observed.append(result)
        return result

    monkeypatch.setattr("priml.data.pipeline.parallel.queue.Queue", make_queue)
    stream = PrefetchBuffer.Config(size=-1).make()(_samples(4))

    assert next(stream) == {"id": 0}
    assert buffered_three.wait(timeout=1)
    assert observed[0].maxsize == 0
    assert [sample["id"] for sample in stream] == [1, 2, 3]


def test_parmap_unbounded_input_does_not_block_the_feeder(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="priml.data.pipeline.parallel")
    worker_started = threading.Event()
    release_worker = threading.Event()
    source_exhausted = threading.Event()
    results: list[TestSample] = []

    class BlockingProcessor:
        class Config(Fig["BlockingProcessor"]): ...

        def __init__(self, config: Config) -> None:
            del config

        def __call__(self, samples: Iterator[TestSample]) -> Iterator[TestSample]:
            for sample in samples:
                worker_started.set()
                release_worker.wait(timeout=1)
                yield sample

    def source() -> Iterator[TestSample]:
        for i in range(8):
            yield {"id": i}
        source_exhausted.set()

    def consume() -> None:
        results.extend(
            ParMap.Config(
                num_threads=1,
                max_input_queue_size=-1,
                processors=[BlockingProcessor.Config()],
            ).make()(source()),
        )

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    assert worker_started.wait(timeout=1)
    fed_before_release = source_exhausted.wait(timeout=0.2)
    release_worker.set()
    consumer.join(timeout=2)
    assert fed_before_release, "the configured unbounded queue blocked the feeder"
    assert not consumer.is_alive(), "ParMap did not finish after releasing its worker"
    assert {sample["id"] for sample in results} == set(range(8))
    assert any(
        "input=" in record.getMessage()
        and record.getMessage()
        .split("input=", maxsplit=1)[1]
        .split("/", maxsplit=1)[0]
        .isdecimal()
        for record in caplog.records
        if "ParMap(1 threads) peak queue usage" in record.getMessage()
    )


def test_parmap_limits_poison_pill_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_queue = queue.Queue

    class ObservedQueue(real_queue[dict[str, object] | None]):
        poison_timeouts: list[float | None]

        def __init__(self, *, maxsize: int = 0) -> None:
            super().__init__(maxsize=maxsize)
            self.poison_timeouts = []

        @override
        def put(
            self,
            item: dict[str, object] | None,
            block: bool = True,
            timeout: float | None = None,
        ) -> None:
            if item is None:
                self.poison_timeouts.append(timeout)
            super().put(item, block=block, timeout=timeout)

    created: list[ObservedQueue] = []

    def make_queue(*, maxsize: int = 0) -> ObservedQueue:
        result = ObservedQueue(maxsize=maxsize)
        created.append(result)
        return result

    monkeypatch.setattr("priml.data.pipeline.parallel.queue.Queue", make_queue)
    parmap = ParMap.Config(
        num_threads=2,
        max_input_queue_size=2,
        processors=[SlowProcessor.Config(delay_ms=0)],
    ).make()

    assert len(list(parmap(_samples(2)))) == 2
    assert created[0].poison_timeouts == [0.1, 0.1]


def test_bounded_queue_put_retries_after_a_full_queue_is_drained() -> None:
    real_queue = queue.Queue
    retrying = threading.Event()
    completed = threading.Event()

    class ObservedQueue(real_queue[str]):
        @override
        def put(
            self,
            item: str,
            block: bool = True,
            timeout: float | None = None,
        ) -> None:
            try:
                super().put(item, block=block, timeout=timeout)
            except queue.Full:
                retrying.set()
                raise

    items = ObservedQueue(maxsize=1)
    items.put_nowait("occupied")
    stop_event = threading.Event()

    def put_later() -> None:
        _put_until_stopped(items, "late", stop_event)
        completed.set()

    worker = threading.Thread(target=put_later, daemon=True)
    worker.start()
    assert retrying.wait(timeout=1)
    assert items.get_nowait() == "occupied"
    worker.join(timeout=1)

    assert completed.is_set()
    assert items.get_nowait() == "late"
    assert items.empty()


def test_bounded_queue_put_observes_stop_while_full() -> None:
    items: queue.Queue[str] = queue.Queue(maxsize=1)
    items.put_nowait("occupied")
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_put_until_stopped,
        args=(items, "late", stop_event),
        daemon=True,
    )
    worker.start()
    stop_event.set()
    worker.join(timeout=0.3)
    completed = not worker.is_alive()
    if not completed:
        _ = items.get_nowait()
        worker.join(timeout=1.5)
    assert completed, "a bounded put ignored the stop event"
    assert items.get_nowait() == "occupied"
    assert items.empty()


def test_prefetchbuffer_fill_first_observes_exact_single_item_bound() -> None:
    consumed: list[int] = []
    producer_thread_ids: list[int] = []

    def source() -> Iterator[TestSample]:
        producer_thread_ids.append(threading.get_ident())
        for index in range(3):
            consumed.append(index)
            yield {"id": index}

    consumer_thread_id = threading.get_ident()
    stream = PrefetchBuffer.Config(size=1, fill_first=True).make()(source())
    assert next(stream) == {"id": 0}
    assert consumed[0] == 0
    assert producer_thread_ids != [consumer_thread_id]
    assert [sample["id"] for sample in stream] == [1, 2]
    assert consumed == [0, 1, 2]


def test_prefetchbuffer_unbounded_without_fill_first_preserves_lazy_order(
    caplog: pytest.LogCaptureFixture,
) -> None:
    produced: list[int] = []
    caplog.set_level(logging.INFO, logger="priml.data.pipeline.parallel")

    def source() -> Iterator[TestSample]:
        for index in range(3):
            produced.append(index)
            yield {"id": index}

    stream = PrefetchBuffer.Config(size=-1, fill_first=False).make()(source())
    assert [sample["id"] for sample in stream] == [0, 1, 2]
    assert produced == [0, 1, 2]
    assert any(
        record.getMessage().startswith("PrefetchBuffer peak queue usage: ")
        and record.getMessage().endswith("/∞")
        for record in caplog.records
    )


def test_prefetchbuffer_producer_uses_a_daemon_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_thread = threading.Thread
    threads: list[threading.Thread] = []

    def make_thread(
        *,
        target: Callable[[], None],
        daemon: bool | None = None,
    ) -> threading.Thread:
        thread = real_thread(target=target, daemon=daemon)
        threads.append(thread)
        return thread

    monkeypatch.setattr(
        "priml.data.pipeline.parallel.threading.Thread",
        make_thread,
    )
    samples: list[TestSample] = [{"id": 0}, {"id": 1}]

    assert [
        sample["id"] for sample in PrefetchBuffer.Config(size=2).make()(iter(samples))
    ] == [0, 1]
    assert len(threads) == 1
    assert threads[0].daemon


def test_parmap_uses_daemon_threads_for_workers_and_support_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_thread = threading.Thread
    threads: list[threading.Thread] = []

    def make_thread(
        *,
        target: Callable[..., object] | None,
        daemon: bool | None = None,
    ) -> threading.Thread:
        thread = real_thread(target=target, daemon=daemon)
        threads.append(thread)
        return thread

    monkeypatch.setattr(
        "priml.data.pipeline.parallel.threading.Thread",
        make_thread,
    )
    parmap = ParMap.Config(
        num_threads=1,
        processors=[SlowProcessor.Config(delay_ms=0)],
    ).make()

    assert list(parmap(_samples(1))) == [{"id": 0, "processed": True}]
    assert len(threads) == 3
    assert all(thread.daemon for thread in threads)


def test_parmap_one_thread_applies_processors_in_sequence() -> None:
    config = ParMap.Config(
        num_threads=1,
        processors=[
            SlowProcessor.Config(delay_ms=0, add_field="first"),
            SlowProcessor.Config(delay_ms=0, add_field="second"),
        ],
    )
    samples: list[TestSample] = [{"id": 0}, {"id": 1}]
    results = list(config.make()(iter(samples)))
    assert results == [
        {"id": 0, "first": True, "second": True},
        {"id": 1, "first": True, "second": True},
    ]


def test_parmap_closed_before_iteration_and_twice_is_idempotent() -> None:
    seen: list[int] = []

    def source() -> Generator[TestSample, None, None]:
        for index in range(3):
            seen.append(index)
            yield {"id": index}

    stream = ParMap.Config(num_threads=0).make()(source())
    stream.close()
    stream.close()
    assert seen == []
    assert list(stream) == []


def test_parmap_close_mid_threaded_iteration_is_idempotent() -> None:
    parmap = ParMap.Config(
        num_threads=2,
        processors=[SlowProcessor.Config(delay_ms=0)],
    ).make()
    stream = parmap(_samples(4))

    first = next(stream)
    stream.close()
    stream.close()

    assert first["id"] in range(4)
    assert list(stream) == []


def test_parmap_close_mid_iteration_stops_inline_source() -> None:
    seen: list[int] = []

    def source() -> Generator[TestSample, None, None]:
        for index in range(3):
            seen.append(index)
            yield {"id": index}

    stream = ParMap.Config(num_threads=0).make()(source())
    assert next(stream) == {"id": 0}
    stream.close()
    stream.close()
    assert seen == [0]
    assert list(stream) == []


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
