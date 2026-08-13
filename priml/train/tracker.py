"""Experiment tracking implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import field
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, override

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
        init_timeout_sec: float = 30.0
        """Seconds W&B may spend waiting for run initialization; 0 keeps default."""
        service_wait_sec: float = 30.0
        """Seconds W&B may wait for its local service; 0 keeps default."""
        allow_startup_failure: bool = True
        """Continue with a no-op tracker if W&B startup raises an exception."""
        flush_interval_sec: float = 15.0
        """Seconds between history-stream transmissions; 0 keeps W&B's default.

        Maps to ``wandb.Settings.x_file_stream_transmit_interval``. The W&B
        client buffers logged history between transmissions; on a long
        training run a large/wedged buffer leaves dashboards empty ("no data")
        while the summary still updates. A modest interval bounds both the
        dashboard lag and how much history one stuck transmission holds back."""
        system_metrics: bool = True
        """Whether W&B collects its built-in system metrics (CPU/GPU/mem/...).

        Maps to ``wandb.Settings.x_disable_stats`` (inverted). Keep on for
        memory / power / network / utilization telemetry; disable only to drop
        every system metric (W&B 0.27 has no per-metric allow-list)."""
        system_metrics_interval_sec: float = 60.0
        """Seconds between system-metric samples; 0 keeps W&B's default (15s).

        Maps to ``wandb.Settings.x_stats_sampling_interval``. The built-in
        system metrics dominate history cardinality; sampling them less often
        cuts the logged-point volume that backs up the dashboard, while still
        capturing memory / power / traffic / utilization trends."""
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
        # ``mode`` is a free-form str in the Config for ergonomics; narrow it to
        # wandb's Literal at this boundary (valid values named in the docstring).
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
        settings_kwargs: dict[str, Any] = {}
        if not config.capture_console:
            settings_kwargs["console"] = "off"
        if config.init_timeout_sec > 0:
            settings_kwargs["init_timeout"] = config.init_timeout_sec
        if config.service_wait_sec > 0:
            settings_kwargs["x_service_wait"] = config.service_wait_sec
        if config.flush_interval_sec > 0:
            settings_kwargs["x_file_stream_transmit_interval"] = (
                config.flush_interval_sec
            )
        if not config.system_metrics:
            settings_kwargs["x_disable_stats"] = True
        elif config.system_metrics_interval_sec > 0:
            settings_kwargs["x_stats_sampling_interval"] = (
                config.system_metrics_interval_sec
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


class TrackerList:
    """Tracker that fans every call out to a set of child trackers.

    Children self-gate (e.g. ``WandbTracker`` no-ops off rank 0, ``FileTracker``
    writes only on rank 0), so ``TrackerList`` forwards on all ranks without any
    central gating.
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
                if (
                    isinstance(child, HasNormalizedWorkingDirPattern)
                    and child.base_dir is None
                ):
                    child.base_dir = self.working_dir
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
        """Forward run notes to every child tracker (notes-less ones ignore)."""
        for tracker in self.trackers.values():
            tracker.log_notes(notes)

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
