"""Short-circuit processor for filtering pipeline.

Provides wrapper that skips already-filtered samples and tracks statistics.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, cast

import logging
import threading
import time

from configgle import Fig, Makeable

from priml.data.custom_types import Processor


if TYPE_CHECKING:
    from collections.abc import Iterator


logger = logging.getLogger(__name__)


__all__ = [
    "FilterStats",
    "ShortCircuitProcessor",
]


class FilterStats:
    """Statistics tracker for filter processors.

    Accumulates drop statistics. Thread-safe for use with multiple threads.
    Each process gets its own instance.
    """

    class Config(Fig["FilterStats"]):
        log_interval_sec: float = 30.0
        """Seconds between amortized drop-statistics logs."""

        top_n_processors: int = 5
        """How many of the heaviest-dropping processors get per-reason detail."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._lock = threading.Lock()
        self.samples_processed = 0
        self.samples_skipped = 0
        self.samples_dropped = 0
        # Track per-processor drop counts: {processor_name: count}.
        self.processor_drops: dict[str, int] = {}
        # Track per-processor-per-reason drop counts: {processor_name: {reason: count}}.
        self.drop_reasons: dict[str, dict[str, int]] = {}
        self.last_log_time = 0.0
        self.start_time = time.time()
        self.last_processed_count = 0

    def record_processed(self) -> None:
        """Increment the processed-sample counter under the lock."""
        with self._lock:
            self.samples_processed += 1

    def record_skipped(self) -> None:
        """Increment the skipped-sample counter under the lock."""
        with self._lock:
            self.samples_skipped += 1

    def record_drop(self, processor_name: str, filter_reasons: list[str]) -> None:
        """Record a dropped sample and its reasons under the lock.

        Args:
          processor_name: Name of the processor that dropped the sample.
          filter_reasons: Reasons the sample was filtered, if any.

        """
        with self._lock:
            self.samples_dropped += 1
            self.processor_drops[processor_name] = (
                self.processor_drops.get(processor_name, 0) + 1
            )
            if not filter_reasons:
                return
            reasons = self.drop_reasons.setdefault(processor_name, {})
            for reason in filter_reasons:
                reasons[reason] = reasons.get(reason, 0) + 1

    def log_statistics(self, force: bool = False) -> None:
        """Log amortized drop statistics.

        Args:
          force: Override the throttle and log regardless of timing.

        """
        if not self._should_log(force):
            return

        self._log_summary()
        self._log_processor_drops()
        self._log_drop_reasons()

        self.last_log_time = time.time()

    def _should_log(self, force: bool) -> bool:
        """Check if enough time has passed to log statistics."""
        if force:
            return True
        current_time = time.time()
        return current_time - self.last_log_time >= self._config.log_interval_sec

    def _log_summary(self) -> None:
        """Log summary statistics."""
        # Calculate throughput.
        samples_since_last = self.samples_processed - self.last_processed_count
        samples_per_sec = (
            samples_since_last / self._config.log_interval_sec
            if self._config.log_interval_sec > 0
            else 0
        )

        # Calculate actual work done (not skipped)
        actually_processed = self.samples_processed - self.samples_skipped

        msg = (
            f"Pipeline stats: Total={self.samples_processed}, "
            f"Processed={actually_processed}, Skipped={self.samples_skipped} (already filtered), "
            f"Dropped={self.samples_dropped} (filtered by any processor), "
            f"Rate={samples_per_sec:.1f} samples/sec"
        )
        logger.info(msg)

        # Update last count for next interval.
        self.last_processed_count = self.samples_processed

    def _log_processor_drops(self) -> None:
        """Log drop counts by processor."""
        if not self.processor_drops:
            return

        drops_summary = ", ".join(
            f"{proc}={count}" for proc, count in sorted(self.processor_drops.items())
        )
        logger.info("Drops by processor: %s", drops_summary)

    def _log_drop_reasons(self) -> None:
        """Log detailed drop reasons for top N processors."""
        if not self.drop_reasons:
            return

        # Get top N dropping processors.
        top_droppers = sorted(
            self.drop_reasons.items(),
            key=lambda x: sum(x[1].values()),
            reverse=True,
        )[: self._config.top_n_processors]

        for proc_name, reasons in top_droppers:
            reasons_summary = ", ".join(
                f"{reason}={count}" for reason, count in sorted(reasons.items())
            )
            logger.info("%s drop reasons: %s", proc_name, reasons_summary)


class ShortCircuitProcessor:
    """Processor wrapper that skips already-filtered samples.

    Used to implement short-circuit behavior: wraps each processor
    so it only processes samples that haven't been filtered yet.
    Filtered samples are passed through unchanged.

    Holds its own per-instance FilterStats tracker for amortized drop
    logging. Stats are not shared across processor instances; each wrapped
    processor accumulates its own counts.
    """

    class Config(Fig["ShortCircuitProcessor"]):
        processor: Makeable[Processor[dict[str, object], dict[str, object]]] | None = (
            None
        )
        """The wrapped processor; required."""

        stats_config: FilterStats.Config | None = None
        """Drop-statistics tracker; ``None`` builds one with its defaults."""

    def __init__(self, config: Config):
        if config.processor is None:
            raise ValueError("Must specify `processor`.")
        self.processor: Processor[dict[str, object], dict[str, object]] = (
            config.processor.make()
        )
        stats_config = config.stats_config or FilterStats.Config()
        self.stats = FilterStats(stats_config)
        self.processor_name: str = type(self.processor).__name__

    def __call__(
        self,
        samples: Iterator[dict[str, object]],
    ) -> Iterator[dict[str, object]]:
        """Drive the wrapped processor once, short-circuiting filtered samples.

        Already-filtered samples bypass the processor entirely and pass through
        in stream order; every non-filtered sample is fed into a single,
        continuous call to the processor so stateful and expand (1->N)
        processors keep their accumulation, windowing, and end-of-stream flush
        contracts. Re-invoking the processor per sample (the old behavior) reset
        that state on every element.
        """
        # Filtered samples discovered while the processor lazily pulls its input
        # are queued here and flushed around each processor output, preserving
        # the original interleaving for one-in-one-out processors.
        bypassed: deque[dict[str, object]] = deque()
        # Drops are counted, not identified by object id: a processor may yield
        # a NEW dict per input (e.g. a field rename), so the output object is
        # not the fed object. Identity matching would mis-record every such
        # pass-through as a drop. The fed count minus the emitted count is the
        # net number the processor filtered out (>= 0 for filter/1:1
        # processors; an expanding 1->N processor simply records no drops).
        fed: list[dict[str, object]] = []
        emitted_count = 0

        for processed_sample in self.processor(self._feed(samples, bypassed, fed)):
            while bypassed:
                yield bypassed.popleft()
            emitted_count += 1
            yield processed_sample

        while bypassed:
            yield bypassed.popleft()

        for _ in range(max(0, len(fed) - emitted_count)):
            # Attribute each drop to the most recent fed sample's reasons; the
            # processor mutates filter_reasons in place before declining to
            # yield, so the last fed sample carries the rejection cause.
            reasons = cast(list[str], fed[-1].get("filter_reasons", [])) if fed else []
            self.stats.record_drop(self.processor_name, reasons)
            logger.debug("%s: dropped a sample", self.processor_name)
        self.stats.log_statistics()

    def _feed(
        self,
        samples: Iterator[dict[str, object]],
        bypassed: deque[dict[str, object]],
        fed: list[dict[str, object]],
    ) -> Iterator[dict[str, object]]:
        """Yield non-filtered samples to the processor; queue filtered ones."""
        for sample in samples:
            self.stats.record_processed()
            if sample.get("filter_reasons"):
                self.stats.record_skipped()
                logger.debug(
                    "%s: skipping filtered sample (key=%s, reasons=%s)",
                    self.processor_name,
                    sample.get("key"),
                    sample.get("filter_reasons"),
                )
                bypassed.append(sample)
                continue
            fed.append(sample)
            yield sample
