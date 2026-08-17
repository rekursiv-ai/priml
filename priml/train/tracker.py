"""Experiment tracking implementations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    cast,
    override,
)

import json
import logging
import os

from configgle import Fig, Makeable
from torch import Tensor
from wrapt import lazy_import

from priml.custom_types import HasNormalizedWorkingDirPattern
from priml.logger import bind_logging_to_current_stdout, replay_buffered_logs
from priml.paths import resolve_working_dir, validated_output_path
from priml.runtime import is_rank_zero
from priml.train.custom_types import TrackerProtocol


wandb = lazy_import("wandb")


if TYPE_CHECKING:
    from typing import Self

    from wandb.sdk.wandb_run import Run as _Run


logger = logging.getLogger(__name__)

_logged_nonscalar_skip = [False]  # config-globals: ignore -- one-shot log guard.
"""One-shot guard (mutable cell) so WandbTracker logs a skip only once."""


def scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Return tracker-safe scalar metrics, dropping non-scalar values.

    A value is scalar when it is a real number or a single-element ``Tensor``;
    everything else (dicts, payloads, multi-element tensors) is dropped so a
    tracker never has to flatten or reject an opaque value.
    """
    scalars: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, Real):
            scalars[key] = float(value)
        elif isinstance(value, Tensor) and value.numel() == 1:
            scalars[key] = float(value.item())
    return scalars


class FileTracker:
    """Write scalar metrics for ONE prefix to a JSON file, overwriting each call.

    Rank-0 only. Captures only calls whose ``prefix`` matches ``capture_prefix``
    (default ``"eval/"``), so the file holds eval scores -- not train-step
    metrics that share the same tracker -- and converges to the final eval's
    value (last write wins). Non-scalar values (e.g. an ``extras`` payload) are
    ignored. A downstream remote-eval reconciler reads this as the
    eval-results file, so it must contain ``eval/*`` keys only.
    """

    class Config(Fig["FileTracker"]):
        base_dir: Path | str | None = None
        """Owner directory supplied during parent finalization."""

        working_dir: Path | str = "/metrics.json"
        """Logical JSON destination; an empty path disables file output."""

        capture_prefix: str = "eval/"
        """Only ``log_metrics`` calls with this exact ``prefix`` are written.

        Defaults to ``eval/`` so train-step logging (``prefix="train/"``) does
        not clobber the eval-results file. Set to ``""`` to capture every call."""

        @override
        def finalize(self) -> Self:
            # An empty ``working_dir`` disables file output; preserve it rather
            # than resolving it to ``Path(".")``.
            if self.working_dir:
                self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Write scalar metrics as flat JSON, atomically replacing the file.

        Only writes when ``prefix == capture_prefix`` (default ``eval/``), so
        train-step metrics on the same tracker do not overwrite the eval file.

        Args:
          metrics: Mapping of metric name to a value; ``prefix`` is prepended to
            each key. Non-scalar values are skipped.
          step: Global step number (unused; the file always holds the latest).
          prefix: String prepended to every metric key.

        """
        del step
        if prefix != self.config.capture_prefix:
            return
        if not is_rank_zero():
            return
        if not self.config.working_dir:
            return
        path = validated_output_path(self.config.working_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        prefixed = {
            f"{prefix}{key}": value for key, value in scalar_metrics(metrics).items()
        }
        tmp.write_text(json.dumps(prefixed, indent=2, sort_keys=True))
        tmp.replace(path)
        logger.info("Wrote metrics to %s", path)

    def log_images(self, key: str, images: list[Any], step: int) -> None:
        """No-op image logging; a metrics file holds scalars only."""
        del key, images, step

    def log_notes(self, notes: str) -> None:
        """No-op; a metrics file has no notes concept."""
        del notes

    def close(self) -> None:
        """No resources to release."""


class _WriterFactory(Protocol):
    """Constructor for SummaryWriter-like optional writer classes."""

    def __call__(self, log_dir: str) -> _Writer: ...


_summary_writer_cls: _WriterFactory | None
try:
    from torch.utils.tensorboard import SummaryWriter

    _summary_writer_cls = SummaryWriter
except ImportError:
    _summary_writer_cls = None


class TensorBoardTracker:
    """TensorBoard experiment tracker."""

    class Config(Fig["TensorBoardTracker"]):
        """TensorBoardTracker configuration."""

        base_dir: Path | str | None = None
        """Owner directory supplied during parent finalization."""

        working_dir: Path | str = "/tensorboard"
        """Logical directory for TensorBoard event files."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        """Initialize TensorBoard tracker."""
        if _summary_writer_cls is None:
            msg = "tensorboard is not installed. Install with: pip install tensorboard"
            raise ImportError(msg)
        self.writer: _Writer | None = _summary_writer_cls(
            str(validated_output_path(config.working_dir))
        )

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Log scalar metrics at a given step.

        Args:
          metrics: Mapping of metric name to a value. ``prefix`` is prepended to
            each key. Non-scalar values (dicts, payloads) are skipped.
          step: Global step number.
          prefix: String prepended to every metric key before logging.

        """
        assert self.writer is not None
        for name, value in scalar_metrics(metrics).items():
            self.writer.add_scalar(f"{prefix}{name}", value, step)

    def log_images(self, key: str, images: list[Any], step: int) -> None:
        """No-op image logging fallback for scalar-only TensorBoard tracker."""
        del key, images, step

    def log_notes(self, notes: str) -> None:
        """No-op; TensorBoard has no run-notes concept."""
        del notes

    def close(self) -> None:
        """Cleanup tracker resources (idempotent)."""
        writer = getattr(self, "writer", None)
        if writer is None:
            return
        writer.close()
        self.writer = None

    def __del__(self) -> None:
        """Best-effort close at GC; safe on partially-constructed instances."""
        self.close()


@dataclass(frozen=True, slots=True, kw_only=True)
class WandbIngestion:
    """How much telemetry the W&B client sends, and how long it waits to start.

    One node because these are a single concern with a single failure mode: the
    dashboard lagging the live run. The client buffers history between
    transmissions, and its built-in system metrics dominate the point volume, so
    the interval and the sampling rate are tuned together or not at all. Every
    value maps to a ``wandb.Settings`` field; ``0`` on a duration defers to
    W&B's own default, while ``system_metrics=False`` disables them outright.
    """

    init_timeout_sec: float = 30.0
    """Seconds W&B may spend waiting for run initialization."""

    service_wait_sec: float = 30.0
    """Seconds W&B may wait for its local service."""

    flush_interval_sec: float = 15.0
    """Seconds between history-stream transmissions.

    A large or wedged buffer leaves dashboards empty ("no data") while the
    summary still updates; a modest interval bounds both the lag and how much
    history one stuck transmission holds back."""

    system_metrics: bool = True
    """Collect W&B's built-in system metrics (CPU/GPU/mem/...).

    Off drops every one of them: W&B 0.27 has no per-metric allow-list."""

    system_metrics_interval_sec: float = 60.0
    """Seconds between system-metric samples; 0 keeps W&B's default (15s)."""


class WandbTracker:
    """Weights & Biases experiment tracker.

    Logs scalars to a single W&B run per job. Rank-safe: only the global
    rank-0 process opens a run; every other rank is a no-op, so an N-rank
    distributed job produces one run, not N. Authentication is read from the
    ``WANDB_API_KEY`` environment variable by ``wandb`` itself.
    """

    class Config(Fig["WandbTracker"]):
        """WandbTracker configuration."""

        project: str = "loop"
        """W&B project the run is logged under."""
        name: str = ""
        """Run name; empty lets W&B auto-generate one."""
        run_id: str = ""
        """Existing W&B run id to resume; empty opens a fresh run.

        Set this to log into a run that already exists (e.g. a standalone eval
        job appending ``eval/*`` to the training run that produced the
        checkpoint) instead of creating a separate run. Resumes with
        ``resume="allow"``, so a non-existent id still starts a run under that
        id rather than erroring."""
        group: str = ""
        """Optional run group (e.g. an experiment family); empty disables it."""
        mode: str = "online"
        """W&B mode: "online", "offline", or "disabled"."""
        base_dir: Path | str | None = None
        """Owner directory supplied during parent finalization."""
        working_dir: Path | str = "/wandb"
        """Logical directory for local W&B run files."""
        capture_console: bool = True
        """Whether W&B captures rank-0 stdout/stderr into its console log.

        Only global rank 0 initializes W&B, so distributed jobs produce one
        W&B console stream instead of one stream per rank. Set false to keep
        W&B on structured metrics only while job logs remain authoritative."""
        replay_startup_logs: bool = False
        """Replay buffered pre-init logs into W&B's wrapped stdout.

        Only meaningful when ``capture_console`` is enabled. The default is off
        to keep tracker setup off the critical path for distributed jobs."""
        allow_startup_failure: bool = True
        """Continue with a no-op tracker if W&B startup raises an exception."""
        ingestion: WandbIngestion = field(default_factory=WandbIngestion)
        """Startup timeouts and history-volume knobs; see :class:`WandbIngestion`."""
        metric_step_metrics: dict[str, str] = field(default_factory=dict[str, str])
        """Metric paths mapped to the metric path used as their W&B x-axis."""
        run_config: dict[str, Any] = field(default_factory=dict[str, Any])
        """Hyperparameters recorded on the run (shown in the W&B config tab)."""
        notes: str = ""
        """Free-text run notes (shown in the W&B run overview).

        Populated by the launcher with the experiment function's docstring --
        hypothesis, changes, and outcome -- so the W&B run says WHAT the
        experiment is and WHY, not just its metrics. Empty leaves the W&B
        default (no notes)."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        """Open a W&B run on rank 0; no-op on every other rank."""
        self._run: _Run | None = None
        if not is_rank_zero():
            return
        mode = cast(
            "Literal['online', 'offline', 'disabled', 'shared']",
            config.mode,
        )
        run_name = config.name or None
        # When ``run_id`` is set, resume that exact run so new metrics append to
        # it (e.g. a standalone eval appending eval/* to the training run).
        # ``resume="allow"`` creates the run if the id does not yet exist rather
        # than failing. A fresh run leaves both None for W&B to auto-generate.
        resume = "allow" if config.run_id else None
        # Tune ingestion volume so the dashboard tracks the live run instead of
        # lagging tens of thousands of steps behind: bound the history-stream
        # transmit cadence, throttle the high-cardinality built-in system
        # metrics (or disable them), all via wandb's experimental settings.
        ingestion = config.ingestion
        settings_kwargs: dict[str, Any] = {}
        if not config.capture_console:
            settings_kwargs["console"] = "off"
        if ingestion.init_timeout_sec > 0:
            settings_kwargs["init_timeout"] = ingestion.init_timeout_sec
        if ingestion.service_wait_sec > 0:
            settings_kwargs["x_service_wait"] = ingestion.service_wait_sec
        if ingestion.flush_interval_sec > 0:
            settings_kwargs["x_file_stream_transmit_interval"] = (
                ingestion.flush_interval_sec
            )
        if not ingestion.system_metrics:
            settings_kwargs["x_disable_stats"] = True
        elif ingestion.system_metrics_interval_sec > 0:
            settings_kwargs["x_stats_sampling_interval"] = (
                ingestion.system_metrics_interval_sec
            )
        settings = wandb.Settings(**settings_kwargs) if settings_kwargs else None
        working_dir = validated_output_path(config.working_dir)
        try:
            working_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "WandbTracker: initializing run "
                "(project=%s, name=%s, mode=%s, capture_console=%s).",
                config.project,
                run_name,
                mode,
                config.capture_console,
            )
            run = wandb.init(
                project=config.project,
                name=run_name,
                id=config.run_id or None,
                resume=resume,
                group=config.group or None,
                mode=mode,
                dir=working_dir,
                config=dict(config.run_config),
                notes=config.notes or None,
                settings=settings,
            )
            self._run = run
            for metric_name, step_metric in config.metric_step_metrics.items():
                run.define_metric(step_metric)
                run.define_metric(metric_name, step_metric=step_metric)
        except Exception:
            if not config.allow_startup_failure:
                raise
            logger.exception("W&B startup failed; continuing with a no-op tracker.")
            self._run = None
            return
        if config.capture_console:
            bind_logging_to_current_stdout()
        logger.info("WandbTracker: run initialized.")
        if config.capture_console and config.replay_startup_logs:
            logger.info("WandbTracker: replaying buffered startup logs.")
            replay_buffered_logs()
            logger.info("WandbTracker: startup log replay complete.")

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Log scalar metrics at ``step``.

        Args:
          metrics: Mapping of metric name to a value; ``prefix`` is prepended to
            each key. Non-scalar values (dicts, payloads, multi-element tensors)
            are skipped -- some callers pass an ``extras`` payload meant for
            other trackers.
          step: Global step number.
          prefix: String prepended to every metric key.

        """
        if self._run is None:
            return
        scalars = scalar_metrics(metrics)
        if not _logged_nonscalar_skip[0]:
            skipped = [name for name in metrics if name not in scalars]
            if skipped:
                logger.debug("WandbTracker skipping non-scalar metrics: %s", skipped)
                _logged_nonscalar_skip[0] = True
        self._run.log(
            {f"{prefix}{name}": value for name, value in scalars.items()},
            step=step,
        )

    def log_images(self, key: str, images: list[Any], step: int) -> None:
        """Log images to W&B at ``step``."""
        if self._run is None:
            return
        self._run.log({key: [wandb.Image(image) for image in images]}, step=step)

    def log_notes(self, notes: str) -> None:
        """Set the W&B run notes, unless an explicit note is already present.

        Rank-0 only (no run elsewhere). An explicitly-configured note (set via
        ``Config.notes`` and passed to ``wandb.init``) wins, so the launcher's
        docstring only fills an otherwise-empty overview.
        """
        if self._run is None or not notes:
            return
        if not self._run.notes:
            self._run.notes = notes

    def close(self) -> None:
        """Finish the W&B run (idempotent)."""
        run = getattr(self, "_run", None)
        if run is None:
            return
        run.finish()
        self._run = None

    def __del__(self) -> None:
        """Best-effort finish at GC; safe on partially-constructed instances."""
        self.close()


class AsyncTracker:
    """Run one explicitly selected tracker on an ordered worker.

    The wrapper owns the thread so transport trackers remain simple and other
    children in a ``TrackerList`` stay synchronous. Metric/image containers are
    copied on submission; their contained values must not be mutated until the
    next ``flush`` or ``close``.

    The measured motivation is an H100 Craftax run whose final training
    interval reached 135,592.5 steps/s while complete pre-evaluation training
    averaged 132,830.7 steps/s, a 2.1% gap. That is an opportunity ceiling that
    includes compilation and other host work, not a tracker-only or async A/B
    speedup; this wrapper isolates the tracker-delivery part for measurement.
    """

    class Config(Fig["AsyncTracker"]):
        """Async tracker wrapper configuration."""

        tracker: Makeable[TrackerProtocol] | None = None
        """Child tracker driven by the worker."""
        enabled: bool = True
        """Use the worker; false preserves synchronous child delivery."""

    def __init__(self, config: Config) -> None:
        """Build the child and, when enabled, its one-worker executor."""
        if config.tracker is None:
            raise ValueError("AsyncTracker requires a child tracker config.")
        self.tracker = config.tracker.make()
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="tracker")
            if config.enabled
            else None
        )
        self._pending: list[Future[None]] = []
        self._closed = False

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Queue a shallow call-time snapshot of one metric batch."""
        payload = dict(metrics) if self._executor is not None else metrics
        self._submit(self.tracker.log_metrics, payload, step, prefix=prefix)

    def log_images(self, key: str, images: list[Any], step: int) -> None:
        """Queue a shallow call-time snapshot of one image batch."""
        payload = list(images) if self._executor is not None else images
        self._submit(self.tracker.log_images, key, payload, step)

    def log_notes(self, notes: str) -> None:
        """Set run notes synchronously before training starts."""
        if self._closed:
            raise RuntimeError("AsyncTracker is closed.")
        self.tracker.log_notes(notes)

    def flush(self) -> None:
        """Wait for every submitted call and surface delivery failures."""
        pending, self._pending = self._pending, []
        for future in pending:
            future.result()

    def close(self) -> None:
        """Drain delivery, stop the worker, and close the child."""
        if self._closed:
            return
        self._closed = True
        try:
            self.flush()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
            self.tracker.close()

    def _submit(
        self,
        function: Callable[..., None],
        /,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Submit without waiting; the single worker preserves call order."""
        if self._closed:
            raise RuntimeError("AsyncTracker is closed.")
        if self._executor is None:
            function(*args, **kwargs)
            return
        self._pending.append(self._executor.submit(function, *args, **kwargs))


class TrackerList:
    """Synchronously fan every call out to independent child trackers.

    Children self-gate (e.g. ``WandbTracker`` no-ops off rank 0, ``FileTracker``
    writes only on rank 0), so this composite forwards on all ranks. Wrap only
    an explicitly thread-safe transport child in ``AsyncTracker``; distributed
    or durable children retain caller-thread ordering.
    """

    class Config(Fig["TrackerList"]):
        """TrackerList configuration."""

        trackers: dict[str, Makeable[TrackerProtocol]] = field(
            default_factory=dict[str, Makeable[TrackerProtocol]],
        )
        """Child trackers, built and driven in insertion order."""
        base_dir: Path | str | None = None
        """Owner directory supplied during parent finalization."""
        working_dir: Path | str = "/"
        """Logical root inherited by child trackers."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            for child in self.trackers.values():
                target = unwrap_tracker_config(child)
                if (
                    isinstance(target, HasNormalizedWorkingDirPattern)
                    and target.base_dir is None
                ):
                    target.base_dir = self.working_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        """Build each child tracker."""
        self.trackers = {name: cfg.make() for name, cfg in config.trackers.items()}

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Forward metrics to every child tracker."""
        for tracker in self.trackers.values():
            tracker.log_metrics(metrics, step, prefix=prefix)

    def log_images(self, key: str, images: list[Any], step: int) -> None:
        """Forward images to every child tracker."""
        for tracker in self.trackers.values():
            tracker.log_images(key, images, step)

    def log_notes(self, notes: str) -> None:
        """Forward run notes to every child tracker."""
        for tracker in self.trackers.values():
            tracker.log_notes(notes)

    def flush(self) -> None:
        """Flush deferred children."""
        for tracker in self.trackers.values():
            flush_tracker(tracker)

    def close(self) -> None:
        """Close every child tracker."""
        for tracker in self.trackers.values():
            tracker.close()


def default_metrics_tracker(
    working_dir: Path | str = "/metrics.json",
) -> TrackerList.Config:
    """Return the standard eval-metrics sink: a FileTracker in a TrackerList.

    The canonical scored-run wiring -- a ``FileTracker`` writes the ``eval/*``
    scalars to ``working_dir`` (default ``/metrics.json``, resolved beneath the
    owning project's directory), wrapped in a ``TrackerList`` so callers can add
    more surfaces (W&B, TensorBoard).
    Centralizes the wiring every scored task otherwise hand-copies. Deliberately
    NOT the default of ``TrainLoop.Config.tracker`` -- generic training runs opt
    in explicitly.

    Args:
      working_dir: Destination JSON path for the ``FileTracker``.

    Returns:
      config: A ``TrackerList.Config`` holding one ``FileTracker`` under the key
        ``"metrics"``.

    """
    metrics = FileTracker.Config()
    metrics.working_dir = working_dir
    tracker = TrackerList.Config()
    tracker.trackers = {"metrics": metrics}
    return tracker


class _Writer(Protocol):
    """Minimal scalar-logging writer interface (satisfied by SummaryWriter)."""

    def add_scalar(self, tag: str, scalar_value: Any, global_step: int) -> None: ...


def unwrap_tracker_config(
    config: Makeable[TrackerProtocol],
) -> Makeable[TrackerProtocol]:
    """Return the one child beneath an asynchronous tracker wrapper."""
    if not isinstance(config, AsyncTracker.Config):
        return config
    if config.tracker is None:
        raise ValueError("AsyncTracker requires a child tracker config.")
    return config.tracker


def flush_tracker(tracker: TrackerProtocol) -> None:
    """Flush a deferred tracker; synchronous trackers need no barrier."""
    if isinstance(tracker, (AsyncTracker, TrackerList)):
        tracker.flush()
