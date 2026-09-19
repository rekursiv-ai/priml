"""Tests for PhaseTimer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import logging
import tempfile
import threading
import time

import pytest

from priml.train.custom_types import CudaEventProtocol, TrackerProtocol
from priml.train.profiler import PhaseTimer, ProfilerSchedule, TorchProfiler


if TYPE_CHECKING:
    import torch


def _phase_timer_config(
    *,
    enabled: bool = False,
    torch_profile: bool = False,
    working_dir: Path | str = "/scratch/profiling",
    heartbeat_interval_sec: float = 20.0,
    cuda_events: bool = False,
) -> PhaseTimer.Config:
    return PhaseTimer.Config(
        enabled=enabled,
        torch_profile=torch_profile,
        working_dir=working_dir,
        heartbeat_interval_sec=heartbeat_interval_sec,
        cuda_events=cuda_events,
    )


class TestPhaseTimerDisabled:
    def test_heartbeat_intervals_come_from_config(self) -> None:
        config = _phase_timer_config(heartbeat_interval_sec=17.0)
        config.fault_dump_interval_sec = 23.0
        timer = config.make()
        assert timer.heartbeat_interval_sec == 17.0
        assert timer.fault_dump_interval_sec == 23.0

    def test_noop_phase(self):
        timer = _phase_timer_config(enabled=False).make()
        with timer.phase("test"):
            pass
        s = timer.summary()
        assert list(s.keys()) == ["total"]

    def test_noop_record(self):
        timer = _phase_timer_config(enabled=False).make()
        timer.record("test", 1.0)
        assert "test" not in timer.summary()

    def test_summary_has_total(self):
        timer = _phase_timer_config(enabled=False).make()
        time.sleep(0.01)
        s = timer.summary()
        assert "total" in s
        assert s["total"] >= 0.01


class TestPhaseTimerEnabled:
    def test_single_phase(self):
        timer = _phase_timer_config(enabled=True).make()
        with timer.phase("work"):
            time.sleep(0.01)
        s = timer.summary()
        assert "work" in s
        assert s["work"] >= 0.01

    def test_accumulating_phases(self):
        timer = _phase_timer_config(enabled=True).make()
        for _ in range(3):
            with timer.phase("work"):
                time.sleep(0.01)
        s = timer.summary()
        assert s["work"] >= 0.03
        assert timer._counts["work"] == 3

    def test_multiple_phases(self):
        timer = _phase_timer_config(enabled=True).make()
        with timer.phase("a"):
            time.sleep(0.01)
        with timer.phase("b"):
            time.sleep(0.01)
        s = timer.summary()
        assert "a" in s
        assert "b" in s

    def test_record(self):
        timer = _phase_timer_config(enabled=True).make()
        timer.record("ext", 1.5)
        timer.record("ext", 0.5)
        assert timer.summary()["ext"] == 2.0
        assert timer._counts["ext"] == 2

    def test_record_and_phase_combine(self):
        timer = _phase_timer_config(enabled=True).make()
        timer.record("work", 1.0)
        with timer.phase("work"):
            time.sleep(0.01)
        s = timer.summary()
        assert s["work"] >= 1.01
        assert timer._counts["work"] == 2

    def test_summary_includes_total(self):
        timer = _phase_timer_config(enabled=True).make()
        time.sleep(0.01)
        s = timer.summary()
        assert s["total"] >= 0.01

    def test_publish_interval_owns_reset_and_tracker_hook(self) -> None:
        """Interval aggregation belongs to the timer, not its caller."""
        timer = _phase_timer_config(enabled=True).make()
        tracker = MagicMock()
        timer.reset_interval()
        timer.record("train_batch_fetch", 1.0)
        timer.record("train_step", 2.0)
        timer.record("train_step", 3.0)

        first = timer.publish_interval(cast(TrackerProtocol, tracker), step=4)
        timer.record("train_step", 7.0)
        second = timer.publish_interval(cast(TrackerProtocol, tracker), step=5)

        assert first["interval_train_batch_fetch_sec"] == 1.0
        assert first["interval_train_batch_fetch_count"] == 1.0
        assert first["interval_train_step_sec"] == 5.0
        assert first["interval_train_step_count"] == 2.0
        assert second["interval_train_step_sec"] == 7.0
        assert second["interval_train_step_count"] == 1.0
        assert "interval_train_batch_fetch_sec" not in second
        assert tracker.log_metrics.call_count == 2
        tracker.log_metrics.assert_any_call(first, 4, prefix="timing/")
        tracker.log_metrics.assert_any_call(second, 5, prefix="timing/")

    def test_publish_summary_owns_machine_readable_tracker_hook(self) -> None:
        """The timer publishes its cumulative totals without caller assembly."""
        timer = _phase_timer_config(enabled=True).make()
        tracker = MagicMock()
        timer.record("train_step", 3.0)
        timer.record("train_step", 2.0)

        summary = timer.publish_summary(cast(TrackerProtocol, tracker), step=8)

        assert summary["train_step_sec"] == 5.0
        assert summary["train_step_count"] == 2.0
        assert summary["total_sec"] >= 0.0
        assert summary["unattributed_sec"] >= 0.0
        tracker.log_metrics.assert_called_once_with(summary, 8, prefix="timing/")


class TestPhaseTimerLogging:
    def test_enter_logs_info(self, caplog: pytest.LogCaptureFixture):
        timer = _phase_timer_config(enabled=True).make()
        with caplog.at_level(logging.INFO), timer.phase("fwd"):
            pass
        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any("[phase] fwd started" in m.message for m in info_msgs)
        assert any("[phase] fwd:" in m.message for m in info_msgs)

    def test_enter_logs_info_when_accounting_disabled(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        """Boundary logging is on even when timing accounting is disabled."""
        timer = _phase_timer_config(enabled=False).make()
        with caplog.at_level(logging.INFO), timer.phase("model_init"):
            pass
        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("[phase] model_init started" in m for m in info_msgs)
        assert any("[phase] model_init:" in m for m in info_msgs)

    def test_every_call_logs_info(self, caplog: pytest.LogCaptureFixture):
        """Repeat calls keep logging at INFO (no DEBUG demotion that hides them)."""
        timer = _phase_timer_config(enabled=True).make()
        with timer.phase("fwd"):
            pass
        caplog.clear()
        with caplog.at_level(logging.INFO), timer.phase("fwd"):
            pass
        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("[phase] fwd started" in m for m in info_msgs)

    def test_log_summary_format(self, caplog: pytest.LogCaptureFixture):
        timer = _phase_timer_config(enabled=True).make()
        with timer.phase("work"):
            time.sleep(0.01)
        with caplog.at_level(logging.INFO):
            timer.log_summary()
        assert any("Phase Timing Summary" in r.message for r in caplog.records)
        assert any("work" in r.message for r in caplog.records)
        assert any("total" in r.message for r in caplog.records)

    def test_log_summary_disabled_noop(self, caplog: pytest.LogCaptureFixture):
        timer = _phase_timer_config(enabled=False).make()
        with caplog.at_level(logging.DEBUG):
            timer.log_summary()
        assert len(caplog.records) == 0


class TestPhaseTimerHeartbeat:
    def test_heartbeat_fires_for_slow_phase(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        timer = _phase_timer_config(heartbeat_interval_sec=0.02).make()
        with caplog.at_level(logging.INFO), timer.phase("slow"):
            time.sleep(0.07)
        msgs = [r.message for r in caplog.records]
        assert any("[phase] still in slow" in m for m in msgs)

    def test_no_heartbeat_for_fast_phase(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        timer = _phase_timer_config(heartbeat_interval_sec=0.5).make()
        with caplog.at_level(logging.INFO), timer.phase("fast"):
            pass
        assert not any("still in" in r.message for r in caplog.records)

    def test_heartbeat_disabled_by_zero_interval(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        timer = _phase_timer_config(heartbeat_interval_sec=0.0).make()
        with caplog.at_level(logging.INFO), timer.phase("slow"):
            time.sleep(0.05)
        assert not any("still in" in r.message for r in caplog.records)

    def test_heartbeat_thread_joined_on_exit(self):
        before = threading.active_count()
        timer = _phase_timer_config(heartbeat_interval_sec=0.01).make()
        with timer.phase("slow"):
            time.sleep(0.03)
        # The re-arming timer thread must be cancelled + joined on exit.
        time.sleep(0.05)
        assert threading.active_count() == before

    def test_heartbeat_thread_joined_on_exception(self):
        before = threading.active_count()
        timer = _phase_timer_config(heartbeat_interval_sec=0.01).make()

        def _raise_inside_phase() -> None:
            with timer.phase("slow"):
                time.sleep(0.03)
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            _raise_inside_phase()
        time.sleep(0.05)
        assert threading.active_count() == before

    def test_non_zero_rank_emits_no_heartbeat(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr("priml.train.profiler.is_rank_zero", lambda: False)
        timer = _phase_timer_config(heartbeat_interval_sec=0.02).make()
        with caplog.at_level(logging.INFO), timer.phase("slow"):
            time.sleep(0.07)
        assert not any("still in" in r.message for r in caplog.records)

    def test_rank_zero_emits_heartbeat(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr("priml.train.profiler.is_rank_zero", lambda: True)
        timer = _phase_timer_config(heartbeat_interval_sec=0.02).make()
        with caplog.at_level(logging.INFO), timer.phase("slow"):
            time.sleep(0.07)
        assert any("[phase] still in slow" in r.message for r in caplog.records)


class TestPhaseTimerRankGating:
    """Phase enter/exit narrative is rank-0 only; errors stay all-ranks."""

    def test_non_zero_rank_suppresses_enter_exit(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr("priml.train.profiler.is_rank_zero", lambda: False)
        timer = _phase_timer_config(enabled=True).make()
        with caplog.at_level(logging.INFO), timer.phase("fwd"):
            pass
        msgs = [r.message for r in caplog.records]
        assert not any("[phase] fwd started" in m for m in msgs)
        assert not any("[phase] fwd:" in m for m in msgs)
        # Accounting still happens on every rank -- only the narration is gated.
        assert "fwd" in timer.summary()

    def test_rank_zero_emits_enter_exit(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr("priml.train.profiler.is_rank_zero", lambda: True)
        timer = _phase_timer_config(enabled=True).make()
        with caplog.at_level(logging.INFO), timer.phase("fwd"):
            pass
        msgs = [r.message for r in caplog.records]
        assert any("[phase] fwd started" in m for m in msgs)
        assert any("[phase] fwd:" in m for m in msgs)

    def test_non_zero_rank_still_logs_errors_inside_phase(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # ERROR-level logging is never gated: each rank's own failure must
        # reach its per-rank file even when the phase narrative is suppressed.
        monkeypatch.setattr("priml.train.profiler.is_rank_zero", lambda: False)
        timer = _phase_timer_config(enabled=True).make()
        with caplog.at_level(logging.INFO), timer.phase("fwd"):
            logging.getLogger("priml.train.profiler").error("rank crash")
        msgs = [r.message for r in caplog.records]
        assert any("rank crash" in m for m in msgs)
        assert not any("[phase] fwd started" in m for m in msgs)


class TestPhaseTimerTorchProfile:
    def test_creates_trace_file(self, monkeypatch: pytest.MonkeyPatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "phase_trace.json.gz"
            averages = MagicMock()
            averages.table.return_value = "ops"
            profiler = MagicMock()
            profiler.key_averages.return_value = averages

            fake_torch = MagicMock()
            fake_torch.cuda.is_available.return_value = False
            fake_torch.profiler.profile.return_value = profiler
            fake_torch.profiler.ProfilerActivity.CPU = "cpu"
            fake_torch.profiler.ProfilerActivity.CUDA = "cuda"
            monkeypatch.setattr("priml.train.profiler.torch", fake_torch)
            timer = _phase_timer_config(
                enabled=True,
                torch_profile=True,
                working_dir=path.parent,
            ).make()
            with timer.phase("work"):
                pass
            timer.log_summary()
            profiler.start.assert_called_once()
            profiler.stop.assert_called_once()
            profiler.export_chrome_trace.assert_called_once_with(str(path))
            assert path.parent.is_dir()

    def test_log_summary_clears_profiler_and_is_idempotent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """COLD-008: a second log_summary() is a no-op, never double-stops."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "phase_trace.json.gz"
            averages = MagicMock()
            averages.table.return_value = "ops"
            profiler = MagicMock()
            profiler.key_averages.return_value = averages

            fake_torch = MagicMock()
            fake_torch.cuda.is_available.return_value = False
            fake_torch.profiler.profile.return_value = profiler
            fake_torch.profiler.ProfilerActivity.CPU = "cpu"
            fake_torch.profiler.ProfilerActivity.CUDA = "cuda"
            monkeypatch.setattr("priml.train.profiler.torch", fake_torch)
            timer = _phase_timer_config(
                enabled=True,
                torch_profile=True,
                working_dir=path.parent,
            ).make()
            with timer.phase("work"):
                pass
            timer.log_summary()
            assert timer._profiler is None
            timer.log_summary()  # `second` call must not re-stop/export.
            profiler.stop.assert_called_once()
            profiler.export_chrome_trace.assert_called_once_with(str(path))

    def test_no_trace_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "phase_trace.json.gz")
            timer = _phase_timer_config(
                enabled=True,
                torch_profile=False,
                working_dir=Path(path).parent,
            ).make()
            with timer.phase("work"):
                pass
            timer.log_summary()
            assert not Path(path).exists()

    def test_no_trace_when_timer_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "phase_trace.json.gz")
            timer = _phase_timer_config(
                enabled=False,
                torch_profile=True,
                working_dir=Path(path).parent,
            ).make()
            timer.log_summary()
            assert not Path(path).exists()


class TestPhaseTimerLogSummaryRankGating:
    """log_summary's timing table + profiler logs are rank-0 only (#395)."""

    def test_non_zero_rank_suppresses_summary(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr("priml.train.profiler.is_rank_zero", lambda: True)
        timer = _phase_timer_config(enabled=True).make()
        with timer.phase("work"):
            pass
        # ``phase`` emits rank-0 "[phase] work started/…s" boundary lines above;
        # clear them so the assertion sees only what ``log_summary`` itself
        # emits on a non-zero rank (must be nothing). Without this the captured
        # phase-enter lines (which contain "work") spuriously fail the check.
        monkeypatch.setattr("priml.train.profiler.is_rank_zero", lambda: False)
        with caplog.at_level(logging.INFO):
            caplog.clear()
            timer.log_summary()
        assert not any("Phase Timing Summary" in r.message for r in caplog.records)
        assert not any("work" in r.message for r in caplog.records)

    def test_rank_zero_emits_summary(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr("priml.train.profiler.is_rank_zero", lambda: True)
        timer = _phase_timer_config(enabled=True).make()
        with timer.phase("work"):
            pass
        with caplog.at_level(logging.INFO):
            timer.log_summary()
        assert any("Phase Timing Summary" in r.message for r in caplog.records)
        assert any("work" in r.message for r in caplog.records)


class TestPhaseTimerCudaEvents:
    def test_cuda_measurement_is_deferred_until_summary(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CUDA timing uses events without synchronizing the hot path."""
        start = MagicMock()
        end = MagicMock()
        start.elapsed_time.return_value = 12.5
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.Event.side_effect = [start, end]
        monkeypatch.setattr("priml.train.profiler.torch", fake_torch)
        timer = _phase_timer_config(enabled=True, cuda_events=True).make()

        with timer.measure("forward_loss"), timer.measure_cuda("gpu"):
            pass

        start.record.assert_called_once()
        end.record.assert_called_once()
        end.synchronize.assert_not_called()
        summary = timer.publish_summary(None, step=1)
        end.synchronize.assert_called_once()
        assert summary["gpu.forward_loss.gpu_sec"] == pytest.approx(0.0125)
        assert summary["gpu.forward_loss.gpu_count"] == 1.0

    def test_record_cuda_events_logs_deferred_summary(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        timer = _phase_timer_config(enabled=True, cuda_events=True).make()
        start = MagicMock()
        end = MagicMock()
        start.elapsed_time.return_value = 12.5

        timer.record_cuda_events(
            "forward",
            cast(CudaEventProtocol, start),
            cast(CudaEventProtocol, end),
        )

        with caplog.at_level(logging.INFO):
            timer.log_summary()
        end.synchronize.assert_called_once()
        start.elapsed_time.assert_called_once_with(end)
        assert any("CUDA Event Timing Summary" in r.message for r in caplog.records)
        assert any("forward" in r.message for r in caplog.records)

    def test_record_cuda_events_invalidates_a_resolved_summary(self) -> None:
        timer = _phase_timer_config(enabled=True, cuda_events=True).make()
        first_start = MagicMock()
        first_end = MagicMock()
        first_start.elapsed_time.return_value = 10.0
        second_start = MagicMock()
        second_end = MagicMock()
        second_start.elapsed_time.return_value = 20.0
        timer.record_cuda_events(
            "eval",
            cast(CudaEventProtocol, first_start),
            cast(CudaEventProtocol, first_end),
        )
        first = timer.publish_summary(None, step=1)

        timer.record_cuda_events(
            "eval",
            cast(CudaEventProtocol, second_start),
            cast(CudaEventProtocol, second_end),
        )
        second = timer.publish_summary(None, step=2)

        assert first["gpu.eval_sec"] == pytest.approx(0.01)
        assert first["gpu.eval_count"] == 1.0
        assert second["gpu.eval_sec"] == pytest.approx(0.03)
        assert second["gpu.eval_count"] == 2.0
        second_end.synchronize.assert_called_once()

    def test_record_cuda_events_disabled_noop(self) -> None:
        timer = _phase_timer_config(enabled=True, cuda_events=False).make()
        start = MagicMock()
        end = MagicMock()

        timer.record_cuda_events(
            "forward",
            cast(CudaEventProtocol, start),
            cast(CudaEventProtocol, end),
        )
        timer.log_summary()

        end.synchronize.assert_not_called()
        start.elapsed_time.assert_not_called()


class TestPhaseTimerTotalCollision:
    def test_phase_named_total_does_not_crash_summary(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """T-033: a phase literally named 'total' must not break log_summary."""
        timer = _phase_timer_config(enabled=True).make()
        with timer.phase("total"):
            time.sleep(0.001)

        with caplog.at_level(logging.INFO):
            timer.log_summary()  # Must not raise KeyError.


class TestTorchProfilingWorkingDir:
    def test_owner_resolves_working_dir(self) -> None:
        config = TorchProfiler.Config(torch_profile=False)
        config.base_dir = "/scratch/runs/study/run-1"

        assert config.make().working_dir == Path("/scratch/runs/study/run-1/profiling")

    def test_disabled_profiler_uses_opinionated_default(self) -> None:
        profiler = TorchProfiler.Config(torch_profile=False).make()

        assert profiler.working_dir == Path("/profiling")

    def test_explicit_path_working_dir_is_literal(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "profiling"
        config = TorchProfiler.Config(
            torch_profile=False,
            working_dir=working_dir,
        )

        assert config.make().working_dir == working_dir


def test_phase_timer_default_trace_path_uses_working_dir() -> None:
    timer = _phase_timer_config().make()

    assert timer._torch_profile_path == Path("/scratch/profiling/phase_trace.json.gz")


def test_phase_timer_working_dir_is_scoped_by_owner() -> None:
    config = PhaseTimer.Config()
    config.base_dir = "/scratch/runs/study/run-1"

    timer = config.make()

    assert timer._torch_profile_path == Path(
        "/scratch/runs/study/run-1/profiling/phase_trace.json.gz",
    )


def _fake_torch(*, cuda: bool) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Build a torch stand-in, its recording profiler, and the profiler's table."""
    averages = MagicMock()
    averages.table.return_value = "ops"
    profiler = MagicMock()
    profiler.key_averages.return_value = averages
    fake = MagicMock()
    fake.cuda.is_available.return_value = cuda
    fake.distributed.is_initialized.return_value = False
    fake.profiler.profile.return_value = profiler
    fake.profiler.ProfilerActivity.CPU = "cpu"
    fake.profiler.ProfilerActivity.CUDA = "cuda"
    fake.profiler.schedule.return_value = "schedule"
    return fake, profiler, averages


class TestTorchProfilerWindow:
    def test_memory_profile_requires_cuda(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake, _, _ = _fake_torch(cuda=False)
        monkeypatch.setattr("priml.train.profiler.torch", fake)
        with pytest.raises(RuntimeError, match="Memory profiling requires CUDA"):
            TorchProfiler.Config(memory_profile=True).make()

    def test_window_records_between_start_and_end_then_exports(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        fake, profiler, averages = _fake_torch(cuda=False)
        monkeypatch.setattr("priml.train.profiler.torch", fake)
        profiling = TorchProfiler.Config(
            torch_profile_start=2,
            torch_profile_end=4,
            profile_cuda=False,
            schedule=ProfilerSchedule(wait=1, warmup=1, active=2),
            working_dir=tmp_path / "prof",
        ).make()
        fake.profiler.schedule.assert_called_once_with(
            wait=1,
            warmup=1,
            active=2,
            repeat=1,
        )
        assert fake.profiler.profile.call_args.kwargs["activities"] == ["cpu"]
        assert fake.profiler.profile.call_args.kwargs["schedule"] == "schedule"

        with caplog.at_level(logging.INFO):
            for step in range(6):
                profiling.on_step_start(step)
                profiling.on_step_end(step)

        profiler.start.assert_called_once()
        assert profiler.step.call_count == 2  # Steps 2 and 3.
        profiler.stop.assert_called_once()
        profiler.export_chrome_trace.assert_called_once_with(
            str(tmp_path / "prof" / "trace_step_4.json.gz"),
        )
        assert averages.table.call_args.kwargs == {
            "sort_by": "self_cpu_time_total",
            "row_limit": 20,
        }
        assert (tmp_path / "prof").is_dir()
        assert profiling.profiler is None
        assert any("Profiler top ops" in r.message for r in caplog.records)

    def test_cuda_activities_sort_by_cuda_time(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        fake, profiler, averages = _fake_torch(cuda=True)
        monkeypatch.setattr("priml.train.profiler.torch", fake)
        profiling = TorchProfiler.Config(
            torch_profile_start=0,
            torch_profile_end=1,
            export_trace=False,
            working_dir=tmp_path,
        ).make()
        assert fake.profiler.profile.call_args.kwargs["activities"] == ["cpu", "cuda"]
        assert fake.profiler.profile.call_args.kwargs["schedule"] is None

        for step in range(2):
            profiling.on_step_start(step)
            profiling.on_step_end(step)

        assert averages.table.call_args.kwargs["sort_by"] == "self_cuda_time_total"
        profiler.export_chrome_trace.assert_not_called()

    def test_memory_window_records_then_dumps_a_snapshot(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        fake, _, _ = _fake_torch(cuda=True)
        monkeypatch.setattr("priml.train.profiler.torch", fake)
        profiling = TorchProfiler.Config(
            torch_profile=False,
            memory_profile=True,
            memory_profile_start=1,
            memory_profile_end=2,
            working_dir=tmp_path / "mem",
        ).make()

        for step in range(3):
            profiling.on_step_start(step)
            profiling.on_step_end(step)

        fake.cuda.memory._record_memory_history.assert_any_call()
        fake.cuda.memory._dump_snapshot.assert_called_once_with(
            str(tmp_path / "mem" / "memory_step_2.pickle"),
        )
        fake.cuda.memory._record_memory_history.assert_called_with(enabled=None)
        assert (tmp_path / "mem").is_dir()

    def test_unprofiled_rank_builds_no_profiler_and_ignores_steps(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake, _, _ = _fake_torch(cuda=False)
        fake.distributed.is_initialized.return_value = True
        fake.distributed.get_rank.return_value = 1
        monkeypatch.setattr("priml.train.profiler.torch", fake)
        profiling = TorchProfiler.Config(ranks=[0], torch_profile_start=0).make()

        assert profiling.profiler is None
        fake.profiler.profile.assert_not_called()
        profiling.on_step_start(0)
        profiling.on_step_end(0)
        fake.cuda.memory._record_memory_history.assert_not_called()

    @pytest.mark.parametrize(
        ("ranks", "rank", "suffix"),
        [
            (None, 3, "_rank_3"),
            ([0], 0, ""),
            ([0, 1], 1, "_rank_1"),
        ],
        ids=["all_ranks", "single_rank", "several_ranks"],
    )
    def test_rank_suffix_disambiguates_multi_rank_traces(
        self,
        monkeypatch: pytest.MonkeyPatch,
        ranks: list[int] | None,
        rank: int,
        suffix: str,
    ) -> None:
        fake, _, _ = _fake_torch(cuda=False)
        fake.distributed.is_initialized.return_value = True
        fake.distributed.get_rank.return_value = rank
        monkeypatch.setattr("priml.train.profiler.torch", fake)
        profiling = TorchProfiler.Config(torch_profile=False, ranks=ranks).make()
        assert profiling._should_profile()
        assert profiling._get_rank_suffix() == suffix


class TestPhaseTimerCudaPaths:
    def test_measure_cuda_is_inert_without_events_enabled(self) -> None:
        timer = _phase_timer_config(enabled=True, cuda_events=False).make()
        with timer.measure_cuda("gpu"):
            pass
        assert timer._cuda_events == {}

    def test_torch_profile_adds_cuda_activity_when_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake, _, _ = _fake_torch(cuda=True)
        monkeypatch.setattr("priml.train.profiler.torch", fake)
        _phase_timer_config(enabled=True, torch_profile=True).make()
        assert fake.profiler.profile.call_args.kwargs["activities"] == ["cpu", "cuda"]

    def test_record_outside_a_phase_uses_the_bare_name(self) -> None:
        timer = _phase_timer_config(enabled=True).make()
        with timer.phase("outer"):
            timer.record("inner", 0.5)
        timer.record("alone", 0.25)
        summary = timer.summary()
        assert summary["outer/inner"] == 0.5
        assert summary["alone"] == 0.25
        assert "inner" not in summary

    def test_measure_is_silent_when_disabled(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        timer = _phase_timer_config(enabled=False).make()
        with caplog.at_level(logging.INFO), timer.measure("quiet"):
            pass
        assert caplog.records == []
        assert "quiet" not in timer.summary()

    def test_publish_is_a_noop_when_disabled(self) -> None:
        timer = _phase_timer_config(enabled=False).make()
        tracker = MagicMock()
        assert timer.publish_interval(cast(TrackerProtocol, tracker), step=1) == {}
        assert timer.publish_summary(cast(TrackerProtocol, tracker), step=1) == {}
        tracker.log_metrics.assert_not_called()

    def test_summary_is_published_once_but_always_returned(self) -> None:
        timer = _phase_timer_config(enabled=True).make()
        tracker = MagicMock()
        timer.record("work", 1.0)
        first = timer.publish_summary(cast(TrackerProtocol, tracker), step=1)
        timer.record("work", 1.0)
        second = timer.publish_summary(cast(TrackerProtocol, tracker), step=2)
        assert first["work_sec"] == 1.0
        assert second["work_sec"] == 2.0
        tracker.log_metrics.assert_called_once_with(first, 1, prefix="timing/")

    def test_publish_is_rank_zero_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("priml.train.profiler.is_rank_zero", lambda: False)
        timer = _phase_timer_config(enabled=True).make()
        tracker = MagicMock()
        timer.record("work", 1.0)
        metrics = timer.publish_interval(cast(TrackerProtocol, tracker), step=1)
        assert metrics["interval_work_sec"] == 1.0
        tracker.log_metrics.assert_not_called()

    def test_cuda_summary_is_resolved_once(self) -> None:
        timer = _phase_timer_config(enabled=True, cuda_events=True).make()
        start = MagicMock()
        end = MagicMock()
        start.elapsed_time.return_value = 4.0
        timer.record_cuda_events(
            "fwd",
            cast(CudaEventProtocol, start),
            cast(CudaEventProtocol, end),
        )
        timer.publish_summary(None, step=1)
        timer.publish_summary(None, step=2)
        end.synchronize.assert_called_once()

    def test_nested_phases_attribute_self_time_to_the_parent(self) -> None:
        timer = _phase_timer_config(enabled=True).make()
        with timer.phase("outer"), timer.phase("inner"):
            time.sleep(0.01)
        summary = timer.summary()
        assert "outer/inner" in summary
        assert summary["outer"] >= summary["outer/inner"]
        assert timer._self_phases["outer"] <= summary["outer"] - summary["outer/inner"]


class TestTorchProfilerCleanup:
    def test_completed_window_cleanup_does_not_stop_again(
        self,
        tmp_path: Path,
    ) -> None:
        """Cleanup must not stop an already-stopped profiler again."""
        config = TorchProfiler.Config(
            torch_profile=False,
            torch_profile_start=5,
            torch_profile_end=6,
            working_dir=str(tmp_path),
        )
        profiling = config.make()
        averages = MagicMock()
        averages.table.return_value = "ops"
        profiler = MagicMock()
        profiler.key_averages.return_value = averages
        profiling.profiler = cast("torch.profiler.profile", profiler)
        profiling._profiler_started = True

        profiling.on_step_end(6)
        profiling.cleanup()

        profiler.stop.assert_called_once()
        profiler.export_chrome_trace.assert_called_once()

    def test_cleanup_stops_running_profiler(self) -> None:
        """T-032: cleanup must stop a profiler still running at training end."""
        config = TorchProfiler.Config(
            torch_profile=True,
            torch_profile_start=5,
            torch_profile_end=10,
            working_dir="/scratch/profiling",
        )
        profiling = config.make()

        # Simulate: profiler started at step 5, training stopped before step 10.
        profiler = MagicMock()
        profiling.profiler = cast("torch.profiler.profile", profiler)
        profiling._profiler_started = True

        profiling.cleanup()

        profiler.stop.assert_called_once()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
