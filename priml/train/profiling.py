"""Profiling implementations for training performance analysis."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Self, cast, override

import contextlib
import logging
import threading
import time

from configgle import Fig
from wrapt import lazy_import

from priml.paths import resolve_working_dir
from priml.runtime import is_rank_zero
from priml.train.custom_types import CudaEventProtocol


torch = lazy_import("torch")
torch_profiler = lazy_import("torch.profiler")


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfilerSchedule:
    """When the torch profiler is warm, recording, and done.

    One node rather than three fields, because ``torch.profiler.schedule``
    takes all three together: a caller setting a window sets every value or
    none, and ``wait=None`` is what says "no schedule" for the whole group.
    """

    wait: int | None = None
    """Steps to skip before warmup; ``None`` disables the schedule."""

    warmup: int = 0
    """Steps traced but discarded, letting caches settle before recording."""

    active: int = 1
    """Steps actually recorded, once per run."""


class TorchProfiling:
    """Torch profiler for CPU/CUDA profiling and memory profiling.

    Combines torch.profiler (CPU/CUDA timeline) and CUDA memory profiling.

    For distributed training, profiling runs only on specified rank(s) to avoid:
    - Storage waste (N ranks x trace files)
    - I/O contention (all ranks writing simultaneously)
    - File collisions (ranks overwriting same files)
    """

    class Config(Fig["TorchProfiling"]):
        """TorchProfiling configuration."""

        torch_profile: bool = True
        """Run the torch profiler over the ``torch_profile_*`` step window."""

        torch_profile_start: int = 5
        """First step the torch profiler records."""

        torch_profile_end: int = 10
        """Step at which the torch profiler stops and exports."""

        profile_cuda: bool = True
        """Include CUDA (CUPTI) activities in the torch profiler.

        CUPTI activity collection hangs on some torch/CUDA stacks (observed
        wedging the profiler window on torch 2.11+cu128 H100, at any rank
        count -- Issue#412). Set False to trace CPU activities only, which
        still gives op-level attribution without the CUPTI hang."""

        with_stack: bool = True
        """Collect Python stack traces for profiler events."""

        record_shapes: bool = False
        """Record operator input shapes."""

        profile_memory: bool = False
        """Record tensor memory allocations."""

        export_trace: bool = True
        """Export a Chrome trace after the profiler stops."""

        schedule: ProfilerSchedule = field(default_factory=ProfilerSchedule)
        """Warmup/record windowing; see :class:`ProfilerSchedule`."""

        memory_profile: bool = False
        """Record a CUDA memory history over the ``memory_profile_*`` window.

        Requires CUDA; construction raises without it rather than silently
        producing no snapshot."""

        memory_profile_start: int = 5
        """First step the CUDA memory history records."""

        memory_profile_end: int = 10
        """Step at which the memory snapshot is dumped and recording stops."""

        base_dir: Path | str | None = None
        """Owner directory supplied during parent finalization."""

        working_dir: Path | str = "/profiling"
        """Logical directory for profiler traces."""

        ranks: list[int] | None = field(default_factory=lambda: [0])
        """Profile these ranks (None = all ranks)."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        """Initialize profiling."""
        if config.memory_profile and not torch.cuda.is_available():
            raise RuntimeError(
                "Memory profiling requires CUDA, but CUDA is not available. "
                "Set memory_profile=False or run on a CUDA-enabled device.",
            )

        # copy fields; don't retain the whole config
        self.torch_profile = config.torch_profile
        self.torch_profile_start = config.torch_profile_start
        self.torch_profile_end = config.torch_profile_end
        self.profile_cuda = config.profile_cuda
        self.with_stack = config.with_stack
        self.record_shapes = config.record_shapes
        self.profile_memory = config.profile_memory
        self.export_trace = config.export_trace
        self.schedule = config.schedule
        self.memory_profile = config.memory_profile
        self.memory_profile_start = config.memory_profile_start
        self.memory_profile_end = config.memory_profile_end
        self.working_dir = Path(config.working_dir)
        self.ranks = config.ranks

        # Setup torch profiler (only if we should profile on this rank).
        # CUDA activities are optional: CUPTI collection hangs on some stacks
        # (Issue#412), so ``profile_cuda=False`` traces CPU only.
        self.profiler: _TorchProfiler | None = None
        if self.torch_profile and self._should_profile():
            activities = [torch_profiler.ProfilerActivity.CPU]
            if self.profile_cuda:
                activities.append(torch_profiler.ProfilerActivity.CUDA)
            schedule = None
            if self.schedule.wait is not None:
                schedule = torch_profiler.schedule(
                    wait=self.schedule.wait,
                    warmup=self.schedule.warmup,
                    active=self.schedule.active,
                    repeat=1,
                )
            self.profiler = _torch_profile()(
                activities=activities,
                with_stack=self.with_stack,
                record_shapes=self.record_shapes,
                profile_memory=self.profile_memory,
                schedule=schedule,
            )

        self._profiler_started = False

    def _should_profile(self) -> bool:
        """Check if profiling should run on current rank.

        Returns:
          should_profile: True if profiling should run on this rank.

        """
        if not torch.distributed.is_initialized():
            return True  # Non-distributed, always profile

        if self.ranks is None:
            return True
        return torch.distributed.get_rank() in self.ranks

    def on_step_start(self, step: int) -> None:
        """Called at the start of each training step."""
        if not self._should_profile():
            return

        if (
            self.profiler
            and not self._profiler_started
            and step == self.torch_profile_start
        ):
            self.profiler.start()
            self._profiler_started = True

        if (
            self.memory_profile
            and step == self.memory_profile_start
            and torch.cuda.is_available()
        ):
            torch.cuda.memory._record_memory_history()  # noqa: SLF001

    def on_step_end(self, step: int) -> None:
        """Called at the end of each training step."""
        if not self._should_profile():
            return

        rank_suffix = self._get_rank_suffix()

        if (
            self.profiler
            and self._profiler_started
            and self.torch_profile_start <= step < self.torch_profile_end
        ):
            self.profiler.step()

        if self.profiler and self._profiler_started and step == self.torch_profile_end:
            profiler = self.profiler
            profiler.stop()
            self._profiler_started = False
            self.profiler = None

            table = profiler.key_averages().table(
                sort_by="self_cuda_time_total"
                if self.profile_cuda
                else "self_cpu_time_total",
                row_limit=20,
            )
            logger.info("Profiler top ops:\n%s", table)
            if self.export_trace:
                trace_path = Path(
                    self.working_dir / f"trace_step_{step}{rank_suffix}.json.gz"
                )
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                profiler.export_chrome_trace(str(trace_path))
                logger.info(f"Saved profiler trace to {trace_path}")

        if (
            self.memory_profile
            and step == self.memory_profile_end
            and torch.cuda.is_available()
        ):
            snapshot_path = Path(
                self.working_dir / f"memory_step_{step}{rank_suffix}.pickle"
            )
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._dump_snapshot(str(snapshot_path))  # noqa: SLF001
            torch.cuda.memory._record_memory_history(enabled=None)  # noqa: SLF001
            logger.info(f"Saved memory snapshot to {snapshot_path}")

    def _get_rank_suffix(self) -> str:
        """Get rank suffix for output filenames.

        Returns:
          suffix: Empty string if profiling single rank, "_rank_N" if profiling multiple.

        """
        if not torch.distributed.is_initialized():
            return ""  # Non-distributed, no suffix needed

        if self.ranks is None:
            return f"_rank_{torch.distributed.get_rank()}"
        if len(self.ranks) == 1:
            return ""  # Single rank, no suffix needed

        return f"_rank_{torch.distributed.get_rank()}"

    def cleanup(self) -> None:
        """Stop a still-running profiler at end of training.

        If training stops before ``torch_profile_end`` the profiler was
        started but never stopped; leaving it running leaks the profiling
        session. Stop it here (best-effort; no trace export for a partial run).
        """
        if self.profiler is not None and self._profiler_started:
            self.profiler.stop()
            self._profiler_started = False


class PhaseTimer:
    """Accumulates wall-clock time for named phases.

    Boundary logging is always on: every ``phase()`` logs ``[phase] X
    started`` on enter and ``[phase] X: Ns`` on exit at INFO, and -- while
    a phase is open -- emits a periodic ``[phase] still in X (Ns elapsed)``
    heartbeat (rank 0 only) so a multi-minute phase proves liveness rather
    than going silent. A slow or stuck startup phase is therefore visible
    without external ``nvidia-smi`` / ``/proc`` forensics.

    Timing accounting (the ``summary()`` table and the optional PyTorch
    profiler trace) is gated by ``enabled``; boundary + heartbeat logging
    is not, so even the default (disabled) configuration narrates startup.
    """

    class Config(Fig["PhaseTimer"]):
        enabled: bool = False
        """Enable phase timing accounting (summary table + torch profiler).

        Boundary enter/exit logging and the liveness heartbeat are always
        on regardless of this flag.
        """

        torch_profile: bool = False
        """Also emit a PyTorch profiler Chrome trace."""

        base_dir: Path | str | None = None
        """Owner directory supplied during parent finalization."""

        working_dir: Path | str = "/profiling"
        """Logical directory for the phase-trace Chrome trace."""

        heartbeat_interval_sec: float = 30.0
        """Seconds between ``still in <phase>`` liveness logs. 0 disables."""

        cuda_events: bool = False
        """Accumulate CUDA event timings and report them at summary time."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self._enabled = config.enabled
        self._torch_profile = config.torch_profile
        self._torch_profile_path = Path(config.working_dir) / "phase_trace.json.gz"
        self._heartbeat_interval_sec = config.heartbeat_interval_sec
        self._cuda_events_enabled = config.cuda_events
        self._phases: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._cuda_events: dict[
            str, list[tuple[CudaEventProtocol, CudaEventProtocol]]
        ] = {}
        self._start_time = time.perf_counter()
        self._profiler: _TorchProfiler | None = None
        self._summary_logged = False

        if self._enabled and self._torch_profile:
            activities = [torch_profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch_profiler.ProfilerActivity.CUDA)
            profiler = _torch_profile()(
                activities=activities,
                with_stack=True,
                acc_events=True,
            )
            profiler.start()
            self._profiler = profiler

    @contextlib.contextmanager
    def phase(self, name: str) -> Generator[None, None, None]:
        # Boundary narrative is rank-0 only: on an N-GPU run every rank would
        # otherwise emit identical enter/exit lines, drowning the console. Each
        # rank's own errors still reach its per-rank file (torchrun redirects).
        narrate = is_rank_zero()
        if narrate:
            logger.info("[phase] %s started", name)
        start = time.perf_counter()
        heartbeat = _PhaseHeartbeat(name, start, self._heartbeat_interval_sec)
        heartbeat.start()
        try:
            yield
        finally:
            # Cancel + join before logging exit so the heartbeat thread can
            # never outlive the phase or fire after the phase is reported done,
            # even when the body raised.
            heartbeat.stop()
            elapsed = time.perf_counter() - start
            if self._enabled:
                self._phases[name] = self._phases.get(name, 0.0) + elapsed
                self._counts[name] = self._counts.get(name, 0) + 1
            if narrate:
                logger.info("[phase] %s: %.4fs", name, elapsed)

    @property
    def cuda_events_enabled(self) -> bool:
        """Return whether CUDA event timing is active."""
        return self._enabled and self._cuda_events_enabled

    def record(self, name: str, elapsed: float) -> None:
        if not self._enabled:
            return
        self._phases[name] = self._phases.get(name, 0.0) + elapsed
        self._counts[name] = self._counts.get(name, 0) + 1

    def record_cuda_events(
        self,
        name: str,
        start: CudaEventProtocol,
        end: CudaEventProtocol,
    ) -> None:
        """Record a CUDA event pair for deferred elapsed-time reporting."""
        if not self.cuda_events_enabled:
            return
        self._cuda_events.setdefault(name, []).append((start, end))

    def summary(self) -> dict[str, float]:
        total = time.perf_counter() - self._start_time
        return {**self._phases, "total": total}

    def log_summary(self) -> None:
        if not self._enabled or self._summary_logged:
            return
        self._summary_logged = True
        # The summary narrative mirrors phase()/heartbeat: rank 0 only. On an
        # N-GPU run every rank would otherwise emit an identical timing table,
        # drowning the rank-0 narrative. The torch profiler only runs on rank 0
        # (TorchProfiling.ranks defaults to [0]), so non-rank-0 has no profiler
        # to stop and ``self._profiler`` is None there.
        narrate = is_rank_zero()
        if self._profiler:
            self._profiler.stop()
            trace_path = Path(self._torch_profile_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._profiler.export_chrome_trace(str(trace_path))
            table = self._profiler.key_averages().table(
                sort_by="self_cpu_time_total",
                row_limit=20,
            )
            # Clear after stop/export so a second log_summary() (error path +
            # shutdown could double-call) is a safe no-op rather than stopping
            # an already-stopped profiler.
            self._profiler = None
            if narrate:
                logger.info("Saved profiler trace to %s", self._torch_profile_path)
                logger.info("Top ops by CPU time:\n%s", table)
        if not narrate:
            return
        # Compute the wall-clock total independently of phase names so a phase
        # literally named "total" doesn't collide with the summary key.
        total = time.perf_counter() - self._start_time
        lines = [
            "Phase Timing Summary",
            f"  {'phase':<20s} {'time':>7s}  {'pct':>7s}  {'count':>5s}",
            f"  {'─' * 20} {'─' * 7}  {'─' * 7}  {'─' * 5}",
        ]
        for name, t in self._phases.items():
            pct = 100 * t / total if total > 0 else 0
            cnt = self._counts[name]
            lines.append(f"  {name:<20s} {t:>6.3f}s  ({pct:4.1f}%)  x{cnt:>3d}")
        lines.append(f"  {'─' * 20} {'─' * 7}  {'─' * 7}  {'─' * 5}")
        lines.append(f"  {'total':<20s} {total:>6.3f}s")
        logger.info("\n".join(lines))
        self._log_cuda_summary()

    def _log_cuda_summary(self) -> None:
        if not self._cuda_events:
            return
        rows: list[tuple[str, float, int]] = []
        total_ms = 0.0
        for name, pairs in self._cuda_events.items():
            elapsed_ms = 0.0
            for start, end in pairs:
                end.synchronize()
                elapsed_ms += start.elapsed_time(end)
            total_ms += elapsed_ms
            rows.append((name, elapsed_ms, len(pairs)))
        lines = [
            "CUDA Event Timing Summary",
            f"  {'phase':<28s} {'gpu_time':>10s}  {'pct':>7s}  {'count':>5s}",
            f"  {'─' * 28} {'─' * 10}  {'─' * 7}  {'─' * 5}",
        ]
        for name, elapsed_ms, count in rows:
            pct = 100 * elapsed_ms / total_ms if total_ms > 0 else 0.0
            lines.append(
                f"  {name:<28s} {elapsed_ms / 1000:>9.3f}s  ({pct:4.1f}%)  x{count:>3d}",
            )
        lines.append(f"  {'─' * 28} {'─' * 10}  {'─' * 7}  {'─' * 5}")
        lines.append(f"  {'total':<28s} {total_ms / 1000:>9.3f}s")
        logger.info("\n".join(lines))


class _PhaseHeartbeat:
    """Rank-0 daemon thread that logs liveness while a phase is open.

    Uses a single re-armed :class:`threading.Timer` rather than a sleep loop
    so that :meth:`stop` cancels a pending tick immediately. The thread is a
    daemon and is always cancelled + joined by ``PhaseTimer.phase``'s
    ``finally`` block, so it can neither outlive the phase nor survive an
    exception inside the phase body.
    """

    def __init__(self, name: str, start: float, interval_sec: float) -> None:
        self._name = name
        self._start = start
        self._interval_sec = interval_sec
        self._lock = threading.Lock()
        self._stopped = False
        self._timer: threading.Timer | None = None

    def start(self) -> None:
        if self._interval_sec <= 0 or not is_rank_zero():
            return
        with self._lock:
            self._arm()

    def stop(self) -> None:
        # Hold the lock to mark stopped and snapshot the live timer atomically
        # against a concurrent re-arm in _tick; cancel + join outside the lock
        # so a tick blocked on the lock cannot deadlock the join.
        with self._lock:
            self._stopped = True
            timer = self._timer
            self._timer = None
        if timer is not None:
            timer.cancel()
            timer.join()

    def _arm(self) -> None:
        timer = threading.Timer(self._interval_sec, self._tick)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _tick(self) -> None:
        elapsed = time.perf_counter() - self._start
        logger.info("[phase] still in %s (%.0fs elapsed)", self._name, elapsed)
        # Re-arm under the lock, but only if stop() has not run; otherwise the
        # freshly-armed timer would outlive the phase.
        with self._lock:
            if not self._stopped:
                self._arm()


class _ProfilerAverages(Protocol):
    def table(self, *, sort_by: str, row_limit: int) -> str: ...


class _TorchProfiler(Protocol):
    def start(self) -> None: ...

    def step(self) -> None: ...

    def stop(self) -> None: ...

    def export_chrome_trace(self, path: str) -> None: ...

    def key_averages(self) -> _ProfilerAverages: ...


class _ProfilerFactory(Protocol):
    def __call__(
        self,
        *,
        activities: Sequence[object],
        with_stack: bool,
        acc_events: bool = False,
        record_shapes: bool = False,
        profile_memory: bool = False,
        schedule: object | None = None,
    ) -> _TorchProfiler: ...


def _torch_profile() -> _ProfilerFactory:
    return cast(_ProfilerFactory, torch_profiler.profile)
