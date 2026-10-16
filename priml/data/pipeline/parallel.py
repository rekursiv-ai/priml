"""Parallel processing for data pipeline using thread pools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import functools
import logging
import queue
import threading

from configgle import Fig, Makeable

from priml.data.custom_types import Processor


if TYPE_CHECKING:
    from collections.abc import Generator, Iterator


logger = logging.getLogger(__name__)


__all__ = [
    "ParMap",
    "PrefetchBuffer",
]


class ParMap:
    """Apply processors to samples in parallel using a thread pool.

    Creates a thread pool that processes samples concurrently. Each thread pulls
    a sample, applies the configured processors, and puts the result in an output queue.

    Useful for parallelizing CPU-bound operations that release Python's GIL
    (e.g., image decoding with PyTurboJPEG, libwebp, PIL).

    IMPORTANT: Results may be yielded out of order due to thread scheduling.

    Thread Safety and RNG State:
        Threads within ParMap share the worker process's global RNG state
        (random.*, torch.rand, etc.). RNG calls interleave across threads in
        an order set by the OS scheduler, so results are NOT deterministic
        even with a fixed seed: both the output order and the per-sample RNG
        draws vary run to run. This is acceptable for training (the
        interleaving does not affect model quality).

        For deterministic RNG call order and output order, set num_threads=0
        to run processors in the calling thread with no thread pool.
        num_threads=1 keeps the output order but still runs the processors on
        a background thread, concurrently with whatever consumes them.

    Each worker drives the processors over ONE continuous stream of the
    samples it pulls, so a stateful or N->1 stage (``Batcher``) accumulates
    within a worker and flushes when that worker's share ends.

    Example:
        # Parallelize image decoding with 8 threads
        cfg.processors = [
            ParMap.Config(
                num_threads=8,
                processors=[
                    CalcResizeDimensions.Config(),
                    GetBytesFromTarHandle.Config(),
                    CropDuringDecodeImage.Config(),
                ]
            ),
            PrefetchBuffer.Config(size=1024),
            Batcher.Config(size=1024),
            CLIPEmbedding.Config(),
        ]

        This creates pipeline parallelism where:
        - 8 threads decode images concurrently (true parallelism)
        - Results buffer in queue
        - Batcher consumes decoded images
        - CLIP embedding runs on batches

    Benefits over single-threaded:
        - True parallel CPU work for GIL-releasing operations
        - Keeps GPU busy by preparing batches faster
        - Composable with PrefetchBuffer for multi-stage parallelism

    """

    class Config(Fig["ParMap"]):
        """Configuration for ParMap processor."""

        num_threads: int = 4
        """Number of worker threads for parallel processing (0 = no threading, runs in main thread)."""

        max_input_queue_size: int = -1
        """Maximum samples waiting for workers (-1 = unbounded)."""

        max_output_queue_size: int = -1
        """Maximum results waiting to be yielded (-1 = unbounded)."""

        processors: list[Makeable[Processor[Any, Any]]] = field(  # pyright: ignore[reportExplicitAny] -- Stages are typed pairwise; a list of them has no expressible element type.
            default_factory=list[Makeable[Processor[Any, Any]]],  # pyright: ignore[reportExplicitAny] -- Stages are typed pairwise; a list of them has no expressible element type.
        )
        """List of processors to apply to each sample."""

    def __init__(self, config: Config):
        if config.num_threads < 0:
            raise ValueError(f"num_threads must be >= 0; got {config.num_threads}.")
        self.num_threads = config.num_threads
        self.max_input_queue_size = config.max_input_queue_size
        self.max_output_queue_size = config.max_output_queue_size
        self.processors: list[Processor[Any, Any]] = [  # pyright: ignore[reportExplicitAny] -- Stages are typed pairwise; a list of them has no expressible element type.
            p.make() for p in config.processors
        ]

    def __call__(
        self,
        samples: Iterator[dict[str, object]],
    ) -> Generator[dict[str, object], None, None]:
        """Process samples in parallel using thread pool.

        Main thread pulls samples from upstream and submits to work queue.
        Worker threads apply processors and put results in output queue.
        Consumer yields from output queue.

        WARNING: Samples may be yielded out of order.

        """
        if self.num_threads == 0 or not self.processors:
            yield from _chain(self.processors, samples)
            return

        run = _ParMapRun(
            processors=self.processors,
            input_queue=queue.Queue(
                maxsize=max(self.max_input_queue_size, 0),
            ),
            output_queue=queue.Queue(
                maxsize=max(self.max_output_queue_size, 0),
            ),
        )
        threads = [
            threading.Thread(target=run.work, daemon=True)
            for _ in range(self.num_threads)
        ]
        for thread in threads:
            thread.start()
        threading.Thread(
            target=functools.partial(run.feed, samples, self.num_threads),
            daemon=True,
        ).start()
        threading.Thread(
            target=functools.partial(run.finish, threads),
            daemon=True,
        ).start()

        try:
            yield from run.drain()
        finally:
            # Reached on exhaustion, an error, or the consumer closing early:
            # every thread polls these, so none outlives the stream.
            run.consumer_closed.set()
            run.stop_event.set()

        input_max = (
            "∞" if self.max_input_queue_size == -1 else str(self.max_input_queue_size)
        )
        output_max = (
            "∞" if self.max_output_queue_size == -1 else str(self.max_output_queue_size)
        )
        logger.info(
            "ParMap(%s threads) peak queue usage: input=%s/%s, output=%s/%s",
            self.num_threads,
            run.peak_input_queue_size,
            input_max,
            run.peak_output_queue_size,
            output_max,
        )


class PrefetchBuffer:
    """Prefetch samples in background thread to overlap I/O and compute.

    Inspired by TensorFlow Datasets (TFDS) prefetch mechanism:
    - tfds uses tf.data.Dataset.prefetch(AUTOTUNE) to overlap producer/consumer
    - This implementation uses a bounded queue with background thread
    - Reduces GPU idle time by preparing next batch while current batch trains

    Reference: tensorflow_datasets/core/dataset_builder.py:1108-1110
    """

    class Config(Fig["PrefetchBuffer"]):
        """Configuration for PrefetchBuffer."""

        size: int = 1_024
        """Buffer size for prefetching samples (-1 = unlimited, queue until input exhausted)."""

        fill_first: bool = False
        """If True, fill buffer before yielding (prevents model thrashing). Also disables background thread when size<0."""

    Input = dict[str, object]
    Output = Input

    def __init__(self, config: Config):
        self.size = config.size
        self.fill_first = config.fill_first

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Prefetch samples in background thread.

        Requires:
          - (any fields) - passes through all sample fields

        Adds:
          - (none) - transparent pass-through with background prefetching

        """
        # size=0 means "no buffering": pass samples straight through with no
        # background thread, equivalent to omitting the PrefetchBuffer.
        if self.size == 0:
            yield from samples
            return

        # Optimization: if fill_first=True and size<0, no thread needed
        # Just eagerly load everything into memory.
        if self.fill_first and self.size <= -1:
            buffer = list(samples)
            logger.info(
                "PrefetchBuffer eagerly loaded %s items (no thread)",
                len(buffer),
            )
            yield from buffer
            return

        run = _PrefetchRun(
            items=queue.Queue(maxsize=max(self.size, 0)),
            fill_bound=self.size if self.fill_first else 0,
        )
        threading.Thread(
            target=functools.partial(run.produce, samples),
            daemon=True,
        ).start()

        try:
            if self.fill_first:
                run.buffer_ready.wait()
                logger.info(
                    "PrefetchBuffer filled: %s/%s items buffered before yielding",
                    run.items.qsize(),
                    self.size,
                )
            yield from run.drain()
        finally:
            run.consumer_closed.set()

        size_str = "∞" if self.size <= -1 else str(self.size)
        logger.info(
            "PrefetchBuffer peak queue usage: %s/%s",
            run.peak_queue_size,
            size_str,
        )


class _End:
    """End-of-stream marker; distinct from any sample, ``None`` included."""


_END: Final = _End()


@dataclass(slots=True, kw_only=True)
class _ParMapRun:
    """Queues, stop signals, and diagnostics shared by one ``ParMap`` call."""

    processors: list[Processor[Any, Any]]  # pyright: ignore[reportExplicitAny] -- Stages are typed pairwise; a list of them has no expressible element type.
    input_queue: queue.Queue[dict[str, object] | _End]
    output_queue: queue.Queue[dict[str, object] | _End]
    errors: list[Exception] = field(default_factory=list[Exception])
    stop_event: threading.Event = field(default_factory=threading.Event)
    """Set on any failure or on close: workers and the feeder stop."""

    consumer_closed: threading.Event = field(default_factory=threading.Event)
    """Set once nobody will read ``output_queue`` again."""

    # Diagnostic only; the races on these counters are benign under the GIL.
    peak_input_queue_size: int = 0
    peak_output_queue_size: int = 0

    def work(self) -> None:
        """Run the processors over ONE continuous stream of this worker's inputs.

        A stateful or N->1 stage (a ``Batcher``) keeps its accumulation and
        end-of-stream flush only when it sees the whole stream; re-invoking it
        per sample handed it one-element streams.
        """
        try:
            for result in _chain(self.processors, self._inputs()):
                _put_until_stopped(self.output_queue, result, self.stop_event)
        except Exception as e:  # noqa: BLE001 -- Worker failures must stop the run and reach the consumer through the shared exception channel.
            self.errors.append(e)
            self.stop_event.set()

    def feed(self, samples: Iterator[dict[str, object]], num_workers: int) -> None:
        """Pull from upstream, distribute to workers, then end every worker.

        Args:
          samples: Upstream stream.
          num_workers: Workers to send an end marker to.

        """
        try:
            for sample in samples:
                if self.stop_event.is_set():
                    return
                _put_until_stopped(self.input_queue, sample, self.stop_event)
                self.peak_input_queue_size = max(
                    self.peak_input_queue_size,
                    self.input_queue.qsize(),
                )
        except Exception as e:  # noqa: BLE001 -- Feeder failures must stop workers and reach the consumer through the shared exception channel.
            self.errors.append(e)
            self.stop_event.set()
        finally:
            # One end marker per worker, each waited for: a worker left without
            # one blocks forever and so does the end of the stream.
            for _ in range(num_workers):
                _put_until_stopped(self.input_queue, _END, self.stop_event)

    def finish(self, workers: list[threading.Thread]) -> None:
        """Wait for every worker, then end the output stream."""
        for worker in workers:
            worker.join()
        _put_until_stopped(self.output_queue, _END, self.consumer_closed)

    def drain(self) -> Iterator[dict[str, object]]:
        """Yield worker results until the end marker; raise the first failure.

        Yields:
          sample: One worker result, in completion order.

        """
        while True:
            item = self.output_queue.get()
            if self.errors:
                raise self.errors[0]
            if isinstance(item, _End):
                return
            self.peak_output_queue_size = max(
                self.peak_output_queue_size,
                self.output_queue.qsize() + 1,
            )
            yield item

    def _inputs(self) -> Iterator[dict[str, object]]:
        """Yield queued samples until this worker's end marker or a stop."""
        while not self.stop_event.is_set():
            try:
                item = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(item, _End):
                return
            yield item


@dataclass(slots=True, kw_only=True)
class _PrefetchRun:
    """Queue, signals, and diagnostics shared by one ``PrefetchBuffer`` call."""

    items: queue.Queue[dict[str, object] | _End]
    fill_bound: int
    """Items buffered before ``buffer_ready``; 0 releases it at once."""

    error: list[Exception] = field(default_factory=list[Exception])
    buffer_ready: threading.Event = field(default_factory=threading.Event)
    consumer_closed: threading.Event = field(default_factory=threading.Event)
    peak_queue_size: int = 0
    """Diagnostic only; the producer/consumer race on it is benign."""

    def produce(self, samples: Iterator[dict[str, object]]) -> None:
        """Buffer upstream samples; forward an upstream failure to the consumer.

        Args:
          samples: Upstream stream.

        """
        try:
            for sample in samples:
                if self.consumer_closed.is_set():
                    return
                _put_until_stopped(self.items, sample, self.consumer_closed)
                self.peak_queue_size = max(self.peak_queue_size, self.items.qsize())
                if self.fill_bound and self.items.qsize() >= self.fill_bound:
                    self.buffer_ready.set()
        except Exception as e:  # noqa: BLE001 -- An upstream failure is re-raised by the consumer, not mistaken for the end of the data.
            self.error.append(e)
        finally:
            self.buffer_ready.set()
            _put_until_stopped(self.items, _END, self.consumer_closed)

    def drain(self) -> Iterator[dict[str, object]]:
        """Yield buffered samples until the end marker; raise an upstream failure.

        Yields:
          sample: One buffered sample, in upstream order.

        """
        while True:
            item = self.items.get()
            if isinstance(item, _End):
                if self.error:
                    raise self.error[0]
                return
            self.peak_queue_size = max(self.peak_queue_size, self.items.qsize() + 1)
            yield item


def _chain(
    processors: list[Processor[Any, Any]],  # pyright: ignore[reportExplicitAny] -- Stages are typed pairwise; a list of them has no expressible element type.
    samples: Iterator[dict[str, object]],
) -> Iterator[dict[str, object]]:
    """Run ``samples`` through every processor in order."""
    stream: Iterator[Any] = samples  # pyright: ignore[reportExplicitAny] -- The stage chain is typed pairwise; the running stream has no single element type.
    for processor in processors:
        stream = processor(stream)
    return stream  # ty: ignore[unsound-return-statement] -- The last stage's output shape is what the pipeline yields.


# A bounded put may block; time out so a dead-worker stop_event is observed
# instead of deadlocking forever.
def _put_until_stopped[T](
    input_queue: queue.Queue[T],
    sample: T,
    stop_event: threading.Event,
) -> None:
    """Put ``sample`` on the queue, retrying until it lands or ``stop_event`` is set."""
    while not stop_event.is_set():
        try:
            input_queue.put(sample, timeout=0.1)
            return
        except queue.Full:
            continue
