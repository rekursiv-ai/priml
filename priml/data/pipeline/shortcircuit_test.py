from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import threading
import time

from configgle import Fig

from priml.data.pipeline.shortcircuit import (
    FilterStats,
    ShortCircuitProcessor,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


class MockProcessor:
    """Mock processor for testing."""

    class Config(Fig["MockProcessor"]):
        drop_keys: list[str] | None = None
        add_reason: str = "test_reason"

    def __init__(self, config: Config):
        self.drop_keys = config.drop_keys or []
        self.add_reason = config.add_reason

    def __call__(
        self,
        samples: Iterator[dict[str, object]],
    ) -> Iterator[dict[str, object]]:
        for sample in samples:
            key = sample.get("key")
            if key in self.drop_keys:
                # Drop this sample (don't yield)
                continue
            # Add filter reason to passed samples.
            filter_reasons_val = sample.get("filter_reasons")
            if isinstance(filter_reasons_val, list):
                filter_reasons = cast(list[str], filter_reasons_val)
            else:
                filter_reasons: list[str] = []
            filter_reasons.append(self.add_reason)
            sample["filter_reasons"] = filter_reasons
            yield sample


class EverySecondProcessor:
    """Stateful processor: emits every 2nd input across the whole stream.

    Holds a cross-sample counter, so it is only correct when driven once over
    the entire stream -- re-instantiating it per sample resets the counter and
    it emits either every sample or none.
    """

    class Config(Fig["EverySecondProcessor"]):
        pass

    def __init__(self, config: Config):
        del config

    def __call__(
        self,
        samples: Iterator[dict[str, object]],
    ) -> Iterator[dict[str, object]]:
        for count, sample in enumerate(samples, start=1):
            if count % 2 == 0:
                yield sample


def test_short_circuit_preserves_stateful_windowing():
    """A windowing (every-2nd) processor yields the correct subset when wrapped.

    Red test for #322: the old per-sample re-drive reset the counter each call,
    so ``count % 2 == 0`` never held and the processor emitted nothing.
    """
    config = ShortCircuitProcessor.Config(processor=EverySecondProcessor.Config())
    processor = ShortCircuitProcessor(config)

    samples: list[dict[str, object]] = [{"key": f"s{i}"} for i in range(6)]

    result = list(processor(iter(samples)))

    # Every 2nd of the 6 non-filtered samples: s1, s3, s5.
    assert [r["key"] for r in result] == ["s1", "s3", "s5"]


def test_short_circuit_stateful_passes_filtered_through_in_order():
    """Filtered samples bypass the stateful processor but keep stream order."""
    config = ShortCircuitProcessor.Config(processor=EverySecondProcessor.Config())
    processor = ShortCircuitProcessor(config)

    samples: list[dict[str, object]] = [
        {"key": "s0"},
        {"key": "s1"},
        {"key": "f", "filter_reasons": ["already"]},
        {"key": "s2"},
        {"key": "s3"},
    ]

    result = list(processor(iter(samples)))

    # Non-filtered stream is s0,s1,s2,s3 -> every 2nd is s1,s3; filtered "f"
    # passes through in its original position relative to the processed samples.
    assert [r["key"] for r in result] == ["s1", "f", "s3"]
    assert processor.stats.samples_skipped == 1


def test_filter_stats_not_singleton():
    """Test FilterStats is NOT a singleton (each process gets its own)."""
    config = FilterStats.Config()
    stats1 = FilterStats(config)
    stats2 = FilterStats(config)
    assert stats1 is not stats2


def test_filter_stats_initialization():
    """Test FilterStats initializes counters."""
    config = FilterStats.Config()
    stats = FilterStats(config)

    assert stats.samples_processed == 0
    assert stats.samples_skipped == 0
    assert stats.samples_dropped == 0
    assert stats.processor_drops == {}
    assert stats.drop_reasons == {}


def test_filter_stats_should_log():
    """Test FilterStats._should_log method."""
    config = FilterStats.Config(log_interval_sec=1.0)
    stats = FilterStats(config)

    # Force should log.
    assert stats._should_log(force=True) is True

    # Before interval.
    stats.last_log_time = time.time()
    assert stats._should_log(force=False) is False

    # After interval.
    stats.last_log_time = time.time() - 2.0
    assert stats._should_log(force=False) is True


def test_filter_stats_public_mutators():
    """FilterStats exposes public locked mutators; callers don't touch _lock."""
    config = FilterStats.Config()
    stats = FilterStats(config)

    stats.record_processed()
    stats.record_processed()
    stats.record_skipped()
    stats.record_drop("ProcA", ["too_small", "blurry"])
    stats.record_drop("ProcA", ["too_small"])

    assert stats.samples_processed == 2
    assert stats.samples_skipped == 1
    assert stats.samples_dropped == 2
    assert stats.processor_drops["ProcA"] == 2
    assert stats.drop_reasons["ProcA"]["too_small"] == 2
    assert stats.drop_reasons["ProcA"]["blurry"] == 1


def test_short_circuit_processor_pass_through():
    """Test ShortCircuitProcessor passes through non-filtered samples."""
    mock_processor_config = MockProcessor.Config(drop_keys=[])
    config = ShortCircuitProcessor.Config(processor=mock_processor_config)
    processor = ShortCircuitProcessor(config)

    samples: list[dict[str, object]] = [
        {"key": "sample1"},
        {"key": "sample2"},
    ]

    result = list(processor(iter(samples)))

    assert len(result) == 2
    assert result[0]["key"] == "sample1"
    assert result[1]["key"] == "sample2"
    # Should have added filter reasons.
    filter_reasons_0 = cast(list[str], result[0]["filter_reasons"])
    filter_reasons_1 = cast(list[str], result[1]["filter_reasons"])
    assert "test_reason" in filter_reasons_0
    assert "test_reason" in filter_reasons_1


def test_short_circuit_processor_drop():
    """Test ShortCircuitProcessor tracks dropped samples."""
    mock_processor_config = MockProcessor.Config(drop_keys=["sample2"])
    config = ShortCircuitProcessor.Config(processor=mock_processor_config)
    processor = ShortCircuitProcessor(config)

    samples: list[dict[str, object]] = [
        {"key": "sample1"},
        {"key": "sample2"},
        {"key": "sample3"},
    ]

    result = list(processor(iter(samples)))

    assert len(result) == 2
    assert result[0]["key"] == "sample1"
    assert result[1]["key"] == "sample3"

    # Check stats.
    assert processor.stats.samples_processed == 3
    assert processor.stats.samples_dropped == 1
    assert processor.stats.processor_drops["MockProcessor"] == 1


def test_short_circuit_processor_skip_filtered():
    """Test ShortCircuitProcessor skips already-filtered samples."""
    mock_processor_config = MockProcessor.Config(drop_keys=[])
    config = ShortCircuitProcessor.Config(processor=mock_processor_config)
    processor = ShortCircuitProcessor(config)

    samples: list[dict[str, object]] = [
        {"key": "sample1"},
        {"key": "sample2", "filter_reasons": ["already_filtered"]},
        {"key": "sample3"},
    ]

    result = list(processor(iter(samples)))

    assert len(result) == 3
    # Sample 2 should not have test_reason added (skipped)
    filter_reasons_0 = cast(list[str], result[0]["filter_reasons"])
    filter_reasons_1 = cast(list[str], result[1]["filter_reasons"])
    filter_reasons_2 = cast(list[str], result[2]["filter_reasons"])
    assert "test_reason" in filter_reasons_0
    assert "test_reason" not in filter_reasons_1
    assert "already_filtered" in filter_reasons_1
    assert "test_reason" in filter_reasons_2

    # Check stats.
    assert processor.stats.samples_processed == 3
    assert processor.stats.samples_skipped == 1
    assert processor.stats.samples_dropped == 0


def test_short_circuit_processor_drop_reasons():
    """Test ShortCircuitProcessor tracks drop reasons."""

    # Create processor that adds reasons before dropping.
    class ReasonProcessor:
        class Config(Fig["ReasonProcessor"]):
            pass

        def __init__(self, config: Config):
            pass

        def __call__(
            self,
            samples: Iterator[dict[str, object]],
        ) -> Iterator[dict[str, object]]:
            for sample in samples:
                key = sample.get("key")
                if key == "drop_me":
                    # Add reason and don't yield.
                    sample["filter_reasons"] = ["reason1", "reason2"]
                    continue
                yield sample

    config = ShortCircuitProcessor.Config(processor=ReasonProcessor.Config())
    processor = ShortCircuitProcessor(config)

    samples: list[dict[str, object]] = [
        {"key": "keep"},
        {"key": "drop_me"},
    ]

    result = list(processor(iter(samples)))

    assert len(result) == 1
    assert processor.stats.samples_dropped == 1
    assert "ReasonProcessor" in processor.stats.drop_reasons
    assert processor.stats.drop_reasons["ReasonProcessor"]["reason1"] == 1
    assert processor.stats.drop_reasons["ReasonProcessor"]["reason2"] == 1


def test_short_circuit_processor_threading():
    """Test ShortCircuitProcessor is thread-safe."""
    mock_processor_config = MockProcessor.Config(drop_keys=["drop"])
    config = ShortCircuitProcessor.Config(processor=mock_processor_config)
    processor = ShortCircuitProcessor(config)

    def process_samples():
        samples: list[dict[str, object]] = [{"key": "keep"}, {"key": "drop"}]
        list(processor(iter(samples)))

    # Run multiple threads.
    threads = [threading.Thread(target=process_samples) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All threads processed 2 samples each.
    assert processor.stats.samples_processed == 20
    assert processor.stats.samples_dropped == 10


def test_short_circuit_processor_logging():
    """Test ShortCircuitProcessor logs debug messages."""
    mock_processor_config = MockProcessor.Config(drop_keys=["sample2"])
    config = ShortCircuitProcessor.Config(processor=mock_processor_config)

    with patch("priml.data.pipeline.shortcircuit.logger") as mock_logger:
        processor = ShortCircuitProcessor(config)

        samples: list[dict[str, object]] = [
            {"key": "sample1"},
            {"key": "sample2"},
        ]

        list(processor(iter(samples)))

        # Check debug log was called for dropped sample.
        assert mock_logger.debug.call_count >= 1


def test_short_circuit_processor_custom_stats_config():
    """Test ShortCircuitProcessor with custom stats config."""
    stats_config = FilterStats.Config(log_interval_sec=5.0, top_n_processors=10)
    mock_processor_config = MockProcessor.Config(drop_keys=[])
    config = ShortCircuitProcessor.Config(
        processor=mock_processor_config,
        stats_config=stats_config,
    )
    processor = ShortCircuitProcessor(config)

    assert processor.stats._config.log_interval_sec == 5.0
    assert processor.stats._config.top_n_processors == 10


def test_filter_stats_log_summary():
    """Test FilterStats._log_summary method."""
    config = FilterStats.Config(log_interval_sec=1.0)
    stats = FilterStats(config)
    stats.samples_processed = 100
    stats.samples_skipped = 20
    stats.samples_dropped = 10
    stats.last_processed_count = 0

    with patch("priml.data.pipeline.shortcircuit.logger") as mock_logger:
        stats._log_summary()

        # Check info log was called with summary.
        assert mock_logger.info.call_count == 1
        log_msg = mock_logger.info.call_args.args[0]
        assert isinstance(log_msg, str)
        assert "Total=100" in log_msg
        assert "Processed=80" in log_msg
        assert "Skipped=20" in log_msg
        assert "Dropped=10" in log_msg


def test_filter_stats_log_processor_drops():
    """Test FilterStats._log_processor_drops method."""
    config = FilterStats.Config()
    stats = FilterStats(config)
    stats.processor_drops = {"Processor1": 10, "Processor2": 5}

    with patch("priml.data.pipeline.shortcircuit.logger") as mock_logger:
        stats._log_processor_drops()

        # Check info log was called.
        assert mock_logger.info.call_count == 1
        # Second arg is the formatted drops string.
        log_msg = mock_logger.info.call_args.args[1]
        assert isinstance(log_msg, str)
        assert "Processor1=10" in log_msg
        assert "Processor2=5" in log_msg


def test_filter_stats_log_drop_reasons():
    """Test FilterStats._log_drop_reasons method."""
    config = FilterStats.Config(top_n_processors=2)
    stats = FilterStats(config)
    stats.drop_reasons = {
        "Processor1": {"reason1": 10, "reason2": 5},
        "Processor2": {"reason3": 8},
        "Processor3": {"reason4": 2},  # Should be excluded (only top 2)
    }

    with patch("priml.data.pipeline.shortcircuit.logger") as mock_logger:
        stats._log_drop_reasons()

        # Should log top 2 processors.
        assert mock_logger.info.call_count == 2


def test_filter_stats_log_statistics_force():
    """Test FilterStats.log_statistics with force=True logs summary."""
    config = FilterStats.Config()
    stats = FilterStats(config)
    stats.samples_processed = 100

    with patch("priml.data.pipeline.shortcircuit.logger") as mock_logger:
        stats.log_statistics(force=True)

        # Should log summary (1 info call for _log_summary)
        assert mock_logger.info.call_count == 1


class NewDictPassThrough:
    """Pass-through processor that yields a NEW dict per input (e.g. a rename).

    Mirrors production processors like ``FieldSetValues``/``FieldRenameKeys``
    that return ``{**sample, ...}``: the output object is not the fed object.
    """

    class Config(Fig["NewDictPassThrough"]):
        pass

    def __init__(self, config: Config) -> None:
        del config

    def __call__(
        self,
        samples: Iterator[dict[str, object]],
    ) -> Iterator[dict[str, object]]:
        for sample in samples:
            yield {**sample, "tagged": True}


def test_short_circuit_does_not_false_drop_new_dict_passthrough():
    """#322: a processor that yields new dicts must record ZERO drops.

    Identity-based drop attribution recorded a drop for every sample because
    the emitted dict was a different object than the fed one. Count-based
    attribution (fed minus emitted) yields zero for a 1:1 pass-through.
    """
    config = ShortCircuitProcessor.Config(processor=NewDictPassThrough.Config())
    processor = ShortCircuitProcessor(config)

    samples: list[dict[str, object]] = [{"key": f"s{i}"} for i in range(4)]
    out = list(processor(iter(samples)))

    assert len(out) == 4
    assert all(s["tagged"] for s in out)
    assert processor.stats.samples_dropped == 0, (
        f"new-dict pass-through falsely recorded "
        f"{processor.stats.samples_dropped} drops"
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
