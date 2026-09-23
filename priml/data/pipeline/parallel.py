"""Parallel processing for data pipeline using thread pools."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Any

import logging
import queue
import threading

from configgle import Fig, Makeable

from priml.data.custom_types import Processor


if TYPE_CHECKING:
    from collections.abc import Iterator


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
        (or 1) to run processors in the calling thread with no thread pool.

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
        self.num_threads = config.num_threads
        self.max_input_queue_size = config.max_input_queue_size
        self.max_output_queue_size = config.max_output_queue_size
        self.processors: list[Processor[Any, Any]] = [  # pyright: ignore[reportExplicitAny] -- Stages are typed pairwise; a list of them has no expressible element type.
            p.make() for p in config.processors
        ]

    def __call__(
        self,
        samples: Iterator[dict[str, object]],
    ) -> Iterator[dict[str, object]]:
        """Process samples in parallel using thread pool.

        Main thread pulls samples from upstream and submits to work queue.
        Worker threads apply processors and put results in output queue.
        Consumer yields from output queue.

        WARNING: Samples may be yielded out of order.

        """
        if self.num_threads < 1 or not self.processors:
            stream: Iterator[Any] = samples  # pyright: ignore[reportExplicitAny] -- The stage chain is typed pairwise; the running stream has no single element type.
            for processor in self.processors:
                stream = processor(stream)
            yield from stream  # ty: ignore[unsound-yield] -- The last stage's output shape is what the pipeline yields.
            return

        input_queue: queue.Queue[dict[str, object] | None] = queue.Queue(
            maxsize=0 if self.max_input_queue_size == -1 else self.max_input_queue_size,
        )
        output_queue: queue.Queue[dict[str, object] | None] = queue.Queue(
            maxsize=0
            if self.max_output_queue_size == -1
            else self.max_output_queue_size,
        )
        exception_holder: list[Exception] = []
        # Set when any worker dies or the consumer stops early, so the feeder
        # stops blocking on a bounded input_queue that no one is draining.
        stop_event = threading.Event()
        peak_input_queue_size = [0]
        peak_output_queue_size = [0]

        def worker() -> None:
            """Worker thread that processes samples from input queue."""
            try:
                while True:
                    sample = input_queue.get()
                    if sample is None:
                        return
                    stream: Iterator[Any] = iter([sample])  # pyright: ignore[reportExplicitAny] -- The stage chain is typed pairwise; the running stream has no single element type.
                    for processor in self.processors:
                        stream = processor(stream)
                    for result in stream:  # pyright: ignore[reportAny] -- The last stage's output shape is what the queue carries.
                        output_queue.put(result)  # pyright: ignore[reportAny] -- Same boundary as the loop binding above.
            except Exception as e:  # noqa: BLE001 -- Feeder failures must stop workers and reach the consumer through the shared exception channel.
                exception_holder.append(e)
                stop_event.set()

        threads = [
            threading.Thread(target=worker, daemon=True)
            for _ in range(self.num_threads)
        ]
        for thread in threads:
            thread.start()

        def feeder() -> None:
            """Pull from upstream and distribute to workers."""
            try:
                for sample in samples:
                    if stop_event.is_set():
                        break
                    _put_until_stopped(input_queue, sample, stop_event)
                    peak_input_queue_size[0] = max(
                        peak_input_queue_size[0],
                        input_queue.qsize(),
                    )
            except Exception as e:  # noqa: BLE001 -- Feeder failures must stop workers and reach the consumer through the shared exception channel.
                exception_holder.append(e)
                stop_event.set()
            finally:
                for _ in range(self.num_threads):
                    # Best-effort poison pills; workers may already be gone.
                    try:
                        input_queue.put(None, timeout=0.1)
                    except queue.Full:
                        break

        feeder_thread = threading.Thread(target=feeder, daemon=True)
        feeder_thread.start()

        def monitor() -> None:
            """Wait for all workers and signal completion."""
            for thread in threads:
                thread.join()
            output_queue.put(None)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

        while True:
            item = output_queue.get()

            if exception_holder:
                stop_event.set()
                raise exception_holder[0]

            if item is None:
                break

            peak_output_queue_size[0] = max(
                peak_output_queue_size[0],
                output_queue.qsize() + 1,
            )

            yield item

        input_max = (
            "∞" if self.max_input_queue_size == -1 else str(self.max_input_queue_size)
        )
        output_max = (
            "∞" if self.max_output_queue_size == -1 else str(self.max_output_queue_size)
        )
        logger.info(
            "ParMap(%s threads) peak queue usage: input=%s/%s, output=%s/%s",
            self.num_threads,
            peak_input_queue_size[0],
            input_max,
            peak_output_queue_size[0],
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
        if self.fill_first and self.size < 0:
            buffer = list(samples)
            logger.info(
                "PrefetchBuffer eagerly loaded %s items (no thread)",
                len(buffer),
            )
            yield from buffer
            return

        q: queue.Queue[dict[str, object] | None] = queue.Queue(
            maxsize=max(self.size, 0),
        )
        # Diagnostic only; the producer/consumer race on this counter is
        # benign under the GIL and never affects correctness.
        peak_queue_size = [0]
        buffer_ready = threading.Event()

        def producer() -> None:
            try:
                for sample in samples:
                    q.put(sample)
                    peak_queue_size[0] = max(peak_queue_size[0], q.qsize())
                    if self.fill_first and self.size >= 0 and q.qsize() >= self.size:
                        buffer_ready.set()
            finally:
                buffer_ready.set()
                q.put(None)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        if self.fill_first:
            buffer_ready.wait()
            size_str = "∞" if self.size < 0 else str(self.size)
            logger.info(
                "PrefetchBuffer filled: %s/%s items buffered before yielding",
                q.qsize(),
                size_str,
            )

        while True:
            item = q.get()
            if item is None:
                break
            peak_queue_size[0] = max(peak_queue_size[0], q.qsize() + 1)
            yield item

        size_str = "∞" if self.size < 0 else str(self.size)
        logger.info(
            "PrefetchBuffer peak queue usage: %s/%s",
            peak_queue_size[0],
            size_str,
        )


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
