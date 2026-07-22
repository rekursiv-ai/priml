"""Tests for PhaseTimer."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import logging
import tempfile
import threading
import time

import pytest

from priml.train.profiling import PhaseTimer, TorchProfiling


def _phase_timer_config(**kwargs: Any) -> PhaseTimer.Config:
    kwargs.setdefault("working_dir", "/scratch/profiling")
    return PhaseTimer.Config(**kwargs)


class TestPhaseTimerDisabled:
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
        monkeypatch.setattr("priml.train.profiling.is_rank_zero", lambda: False)
        timer = _phase_timer_config(heartbeat_interval_sec=0.02).make()
        with caplog.at_level(logging.INFO), timer.phase("slow"):
            time.sleep(0.07)
        assert not any("still in" in r.message for r in caplog.records)

    def test_rank_zero_emits_heartbeat(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr("priml.train.profiling.is_rank_zero", lambda: True)
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
        monkeypatch.setattr("priml.train.profiling.is_rank_zero", lambda: False)
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
        monkeypatch.setattr("priml.train.profiling.is_rank_zero", lambda: True)
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
        monkeypatch.setattr("priml.train.profiling.is_rank_zero", lambda: False)
        timer = _phase_timer_config(enabled=True).make()
        with caplog.at_level(logging.INFO), timer.phase("fwd"):
            logging.getLogger("priml.train.profiling").error("rank crash")
        msgs = [r.message for r in caplog.records]
        assert any("rank crash" in m for m in msgs)
        assert not any("[phase] fwd started" in m for m in msgs)


class TestPhaseTimerTorchProfile:
    def test_creates_trace_file(self, monkeypatch: pytest.MonkeyPatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "phase_trace.json.gz"
            profiler = MagicMock()
            profiler.key_averages.return_value.table.return_value = "ops"

            fake_profiler_module = MagicMock()
            fake_profiler_module.profile.return_value = profiler
            fake_profiler_module.ProfilerActivity.CPU = "cpu"
            fake_profiler_module.ProfilerActivity.CUDA = "cuda"
            fake_torch = MagicMock()
            fake_torch.cuda.is_available.return_value = False

            monkeypatch.setattr(
                "priml.train.profiling.torch_profiler",
                fake_profiler_module,
            )
            monkeypatch.setattr("priml.train.profiling.torch", fake_torch)
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
            profiler = MagicMock()
            profiler.key_averages.return_value.table.return_value = "ops"

            fake_profiler_module = MagicMock()
            fake_profiler_module.profile.return_value = profiler
            fake_profiler_module.ProfilerActivity.CPU = "cpu"
            fake_profiler_module.ProfilerActivity.CUDA = "cuda"
            fake_torch = MagicMock()
            fake_torch.cuda.is_available.return_value = False

            monkeypatch.setattr(
                "priml.train.profiling.torch_profiler",
                fake_profiler_module,
            )
            monkeypatch.setattr("priml.train.profiling.torch", fake_torch)
            timer = _phase_timer_config(
                enabled=True,
                torch_profile=True,
                working_dir=path.parent,
            ).make()
            with timer.phase("work"):
                pass
            timer.log_summary()
            assert timer._profiler is None
            timer.log_summary()  # second call must not re-stop/export
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
        monkeypatch.setattr("priml.train.profiling.is_rank_zero", lambda: True)
        timer = _phase_timer_config(enabled=True).make()
        with timer.phase("work"):
            pass
        # ``phase`` emits rank-0 "[phase] work started/…s" boundary lines above;
        # clear them so the assertion sees only what ``log_summary`` itself
        # emits on a non-zero rank (must be nothing). Without this the captured
        # phase-enter lines (which contain "work") spuriously fail the check.
        monkeypatch.setattr("priml.train.profiling.is_rank_zero", lambda: False)
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
        monkeypatch.setattr("priml.train.profiling.is_rank_zero", lambda: True)
        timer = _phase_timer_config(enabled=True).make()
        with timer.phase("work"):
            pass
        with caplog.at_level(logging.INFO):
            timer.log_summary()
        assert any("Phase Timing Summary" in r.message for r in caplog.records)
        assert any("work" in r.message for r in caplog.records)


class TestPhaseTimerCudaEvents:
    def test_record_cuda_events_logs_deferred_summary(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        timer = _phase_timer_config(enabled=True, cuda_events=True).make()
        start = MagicMock()
        end = MagicMock()
        start.elapsed_time.return_value = 12.5

        timer.record_cuda_events("forward", start, end)

        with caplog.at_level(logging.INFO):
            timer.log_summary()
        end.synchronize.assert_called_once()
        start.elapsed_time.assert_called_once_with(end)
        assert any("CUDA Event Timing Summary" in r.message for r in caplog.records)
        assert any("forward" in r.message for r in caplog.records)

    def test_record_cuda_events_disabled_noop(self) -> None:
        timer = _phase_timer_config(enabled=True, cuda_events=False).make()
        start = MagicMock()
        end = MagicMock()

        timer.record_cuda_events("forward", start, end)
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
            timer.log_summary()  # must not raise KeyError


class TestTorchProfilingWorkingDir:
    def test_owner_resolves_working_dir(self) -> None:
        config = TorchProfiling.Config(torch_profile=False)
        config.base_dir = "/scratch/runs/study/run-1"

        assert config.make().working_dir == Path("/scratch/runs/study/run-1/profiling")

    def test_disabled_profiler_uses_opinionated_default(self) -> None:
        profiling = TorchProfiling.Config(torch_profile=False).make()

        assert profiling.working_dir == Path("/profiling")

    def test_working_dir_rejected_when_trace_is_written(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checkout = tmp_path / "checkout"
        checkout.mkdir()
        (checkout / ".git").mkdir()
        monkeypatch.chdir(checkout)
        profiling = TorchProfiling.Config(
            torch_profile=False,
            torch_profile_end=1,
            working_dir=".",
        ).make()
        profiling.profiler = MagicMock()
        profiling._profiler_started = True

        with pytest.raises(ValueError, match="outside Git checkout"):
            profiling.on_step_end(1)

    def test_explicit_path_working_dir_is_literal(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "profiling"
        config = TorchProfiling.Config(
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
        "/scratch/runs/study/run-1/profiling/phase_trace.json.gz"
    )


def test_phase_timer_trace_path_rejects_git_checkout_when_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    monkeypatch.chdir(checkout)
    timer = PhaseTimer.Config(
        enabled=True,
        torch_profile=False,
        working_dir=".",
    ).make()
    timer._profiler = MagicMock()

    with pytest.raises(
        ValueError,
        match="runtime output path must be outside Git checkout",
    ):
        timer.log_summary()


class TestTorchProfilingCleanup:
    def test_completed_window_cleanup_does_not_stop_again(
        self,
        tmp_path: Path,
    ) -> None:
        """Cleanup must not stop an already-stopped profiler again."""
        config = TorchProfiling.Config(
            torch_profile=False,
            torch_profile_start=5,
            torch_profile_end=6,
            working_dir=str(tmp_path),
        )
        profiling = config.make()
        profiler = MagicMock()
        profiler.key_averages.return_value.table.return_value = "ops"
        profiling.profiler = profiler
        profiling._profiler_started = True

        profiling.on_step_end(6)
        profiling.cleanup()

        profiler.stop.assert_called_once()
        profiler.export_chrome_trace.assert_called_once()

    def test_cleanup_stops_running_profiler(self) -> None:
        """T-032: cleanup must stop a profiler still running at training end."""
        config = TorchProfiling.Config(
            torch_profile=True,
            torch_profile_start=5,
            torch_profile_end=10,
            working_dir="/scratch/profiling",
        )
        profiling = config.make()

        # Simulate: profiler started at step 5, training stopped before step 10.
        profiler = MagicMock()
        profiling.profiler = profiler
        profiling._profiler_started = True

        profiling.cleanup()

        profiler.stop.assert_called_once()
