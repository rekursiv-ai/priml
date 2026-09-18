"""Experiment tracking implementations."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import (
    TYPE_CHECKING,
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


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Self

    from wandb.sdk.wandb_run import Run

    import wandb
else:
    wandb = lazy_import("wandb")


logger = logging.getLogger(__name__)


def scalar_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    """Return real numbers and one-element tensors as floats.

    Args:
      metrics: Mixed metric values.

    Returns:
      scalars: Values accepted by external trackers.

    """
    scalars: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, Real):
            scalars[key] = float(value)
        elif isinstance(value, Tensor) and value.numel() == 1:
            scalars[key] = float(value.item())
    return scalars


class FileTracker:
    """Write the latest scalar metrics for one prefix to a JSON file on rank zero.

    Only calls whose ``prefix`` equals ``capture_prefix`` are written; the
    default ``"eval/"`` keeps training metrics out of the eval-results file.
    """

    class Config(Fig["FileTracker"]):
        base_dir: Path | str | None = None
        """Owner directory supplied during parent finalization."""

        working_dir: Path | str = "/metrics.json"
        """Logical JSON destination; an empty path disables file output."""

        capture_prefix: str = "eval/"
        """Exact prefix required by ``log_metrics``; empty captures every call."""

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
        metrics: Mapping[str, object],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Atomically replace the JSON file with scalar metrics for one prefix.

        Args:
          metrics: Mixed metric values; non-scalars are skipped.
          step: Ignored; the file stores the latest accepted values.
          prefix: Must equal ``capture_prefix`` for the write to occur.

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

    def log_images(self, key: str, images: list[object], step: int) -> None:
        """Ignore image logging."""
        del key, images, step

    def log_notes(self, notes: str) -> None:
        """Ignore run notes."""
        del notes

    def close(self) -> None:
        """Close the tracker; this implementation has no resources."""


class _Writer(Protocol):
    """Minimal scalar-logging writer interface."""

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None: ...

    def close(self) -> None: ...


class _WriterFactory(Protocol):
    """Factory for optional SummaryWriter-like classes."""

    def __call__(self, log_dir: str) -> _Writer: ...


_summary_writer_cls: _WriterFactory | None = None


class TensorBoardTracker:
    """Log scalar metrics to optional TensorBoard event files."""

    class Config(Fig["TensorBoardTracker"]):
        """TensorBoard tracker configuration."""

        base_dir: Path | str | None = None
        """Owner directory supplied during parent finalization."""

        working_dir: Path | str = "/tensorboard"
        """Logical directory for TensorBoard event files."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        """Initialize TensorBoard, raising if its optional dependency is unavailable."""
        writer_cls = _summary_writer_cls
        if writer_cls is None:
            try:
                from torch.utils.tensorboard import (  # noqa: PLC0415 -- TensorBoard is optional and loaded only when selected.
                    SummaryWriter,
                )
            except ImportError as error:
                msg = (
                    "tensorboard is not installed. "
                    "Install with: pip install tensorboard"
                )
                raise ImportError(msg) from error
            writer_cls = cast(_WriterFactory, SummaryWriter)
        self.writer: _Writer | None = writer_cls(
            str(validated_output_path(config.working_dir)),
        )

    def log_metrics(
        self,
        metrics: Mapping[str, object],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Log scalar metrics at ``step``, prepending ``prefix`` to each key."""
        if self.writer is None:
            raise ValueError("Expected self.writer is not None.")
        for name, value in scalar_metrics(metrics).items():
            self.writer.add_scalar(f"{prefix}{name}", value, step)

    def log_images(self, key: str, images: list[object], step: int) -> None:
        """Ignore image logging."""
        del key, images, step

    def log_notes(self, notes: str) -> None:
        """Ignore run notes."""
        del notes

    def close(self) -> None:
        """Close the TensorBoard writer, if open."""
        if "writer" not in self.__dict__ or self.writer is None:
            return
        self.writer.close()
        self.writer = None

    def __del__(self) -> None:
        """Best-effort close during garbage collection."""
        self.close()


@dataclass(frozen=True, slots=True, kw_only=True)
class WandbIngestion:
    """Configure W&B startup, history flushing, and system-metric ingestion."""

    init_timeout_sec: float = 30.0
    """Seconds W&B may spend waiting for run initialization."""

    service_wait_sec: float = 30.0
    """Seconds W&B may wait for its local service."""

    flush_interval_sec: float = 15.0
    """Seconds between history-stream transmissions; zero uses W&B's default."""

    system_metrics: bool = True
    """Collect W&B's built-in system metrics."""

    system_metrics_interval_sec: float = 60.0
    """Seconds between system-metric samples; zero uses W&B's default."""


class WandbTracker:
    """Log one W&B run per job from global rank zero."""

    class Config(Fig["WandbTracker"]):
        """W&B tracker configuration."""

        project: str = "loop"
        """W&B project for the run."""

        name: str = ""
        """Run name; empty lets W&B auto-generate one."""

        run_id: str = ""
        """Run id to resume; empty opens a fresh run. Uses ``resume="allow"``."""

        group: str = ""
        """Optional run group (e.g. an experiment family); empty disables it."""

        mode: str = "online"
        """W&B mode: "online", "offline", or "disabled"."""

        base_dir: Path | str | None = None
        """Owner directory supplied during parent finalization."""

        working_dir: Path | str = "/wandb"
        """Logical directory for local W&B run files."""

        capture_console: bool = True
        """Capture global rank-zero stdout and stderr in the W&B console log."""

        replay_startup_logs: bool = False
        """Replay buffered startup logs when console capture is enabled."""

        allow_startup_failure: bool = True
        """Continue with a no-op tracker when W&B startup fails."""

        ingestion: WandbIngestion = field(default_factory=WandbIngestion)
        """Startup timeouts and history-volume knobs; see :class:`WandbIngestion`."""

        metric_step_metrics: dict[str, str] = field(default_factory=dict[str, str])
        """Metric paths mapped to the metric path used as their W&B x-axis."""

        run_config: dict[str, object] = field(default_factory=dict[str, object])
        """Hyperparameters recorded on the run (shown in the W&B config tab)."""

        notes: str = ""
        """Free-text notes shown in the W&B run overview."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    # A class default rather than an ``__init__`` assignment: tests build the
    # tracker with ``__new__`` and a fake run, bypassing ``__init__``.
    _logged_nonscalar_skip = False

    def __init__(self, config: Config) -> None:
        """Open the W&B run on rank zero; other ranks remain no-ops."""
        self._run: Run | None = None
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
        settings_kwargs: dict[str, object] = {}
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
        settings = (
            cast("Callable[..., object]", wandb.Settings)(**settings_kwargs)
            if settings_kwargs
            else None
        )
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
            run = cast("Callable[..., Run]", wandb.init)(
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
        metrics: Mapping[str, object],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Log scalar metrics at ``step``, prepending ``prefix`` to each key.

        Args:
          metrics: Mixed metric values; non-scalars are skipped.
          step: Global step associated with the metrics.
          prefix: Prefix prepended to each metric name.

        """
        if self._run is None:
            return
        scalars = scalar_metrics(metrics)
        if not self._logged_nonscalar_skip:
            skipped = [name for name in metrics if name not in scalars]
            if skipped:
                logger.debug("WandbTracker skipping non-scalar metrics: %s", skipped)
                self._logged_nonscalar_skip = True
        self._run.log(
            {f"{prefix}{name}": value for name, value in scalars.items()},
            step=step,
        )

    def log_images(self, key: str, images: list[object], step: int) -> None:
        """Log images to W&B at ``step`` under ``key``."""
        if self._run is None:
            return
        image = cast("Callable[[object], object]", wandb.Image)
        self._run.log({key: [image(item) for item in images]}, step=step)

    def log_notes(self, notes: str) -> None:
        """Set notes only when the run has no configured notes."""
        if self._run is None or not notes:
            return
        if not self._run.notes:
            self._run.notes = notes

    def close(self) -> None:
        """Finish the W&B run."""
        if "_run" not in self.__dict__ or self._run is None:
            return
        self._run.finish()
        self._run = None

    def __del__(self) -> None:
        """Best-effort finish during garbage collection."""
        self.close()


class AsyncTracker:
    """Deliver one tracker asynchronously on a single ordered worker.

    Metric and image containers are shallow-copied on submission; do not mutate
    their contained values until the next ``flush`` or ``close``.
    """

    class Config(Fig["AsyncTracker"]):
        """Asynchronous tracker wrapper configuration."""

        tracker: Makeable[TrackerProtocol] | None = None
        """Child tracker driven by the worker."""

        enabled: bool = True
        """Use the worker; false preserves synchronous child delivery."""

    def __init__(self, config: Config) -> None:
        """Build the child and optionally start its single worker."""
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
        metrics: Mapping[str, object],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Queue a shallow snapshot of one metric batch."""
        payload = dict(metrics) if self._executor is not None else metrics
        self._submit(self.tracker.log_metrics, payload, step, prefix=prefix)

    def log_images(self, key: str, images: list[object], step: int) -> None:
        """Queue a shallow snapshot of one image batch."""
        payload = list(images) if self._executor is not None else images
        self._submit(self.tracker.log_images, key, payload, step)

    def log_notes(self, notes: str) -> None:
        """Set run notes synchronously before training starts."""
        if self._closed:
            raise RuntimeError("AsyncTracker is closed.")
        self.tracker.log_notes(notes)

    def flush(self) -> None:
        """Wait for submitted calls in order and surface delivery failures."""
        pending, self._pending = self._pending, []
        for future in pending:
            future.result()

    def close(self) -> None:
        """Flush pending calls, stop the worker, and close the child."""
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
    """Forward calls synchronously to child trackers in insertion order.

    Children handle rank gating themselves; ``close`` and ``flush`` follow the
    same order. Use ``AsyncTracker`` only for a thread-safe transport child.
    """

    class Config(Fig["TrackerList"]):
        """Tracker list configuration."""

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
        """Build children in configuration insertion order."""
        self.trackers = {name: cfg.make() for name, cfg in config.trackers.items()}

    def log_metrics(
        self,
        metrics: Mapping[str, object],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Forward metrics to each child in insertion order."""
        for tracker in self.trackers.values():
            tracker.log_metrics(metrics, step, prefix=prefix)

    def log_images(self, key: str, images: list[object], step: int) -> None:
        """Forward images to each child in insertion order."""
        for tracker in self.trackers.values():
            tracker.log_images(key, images, step)

    def log_notes(self, notes: str) -> None:
        """Forward run notes to each child in insertion order."""
        for tracker in self.trackers.values():
            tracker.log_notes(notes)

    def flush(self) -> None:
        """Flush deferred children in insertion order."""
        for tracker in self.trackers.values():
            flush_tracker(tracker)

    def close(self) -> None:
        """Close children in insertion order."""
        for tracker in self.trackers.values():
            tracker.close()


def flush_tracker(tracker: TrackerProtocol) -> None:
    """Flush a deferred tracker; synchronous trackers need no barrier."""
    if isinstance(tracker, (AsyncTracker, TrackerList)):
        tracker.flush()


def unwrap_tracker_config(
    config: Makeable[TrackerProtocol],
) -> Makeable[TrackerProtocol]:
    """Return the child config beneath an asynchronous wrapper."""
    if not isinstance(config, AsyncTracker.Config):
        return config
    if config.tracker is None:
        raise ValueError("AsyncTracker requires a child tracker config.")
    return config.tracker


def default_metrics_tracker(
    working_dir: Path | str = "/metrics.json",
) -> TrackerList.Config:
    """Return a tracker list containing the standard eval-metrics file sink.

    Args:
      working_dir: JSON destination for captured ``eval/`` metrics.

    Returns:
      tracker: A ``TrackerList.Config`` containing the file sink under
        ``"metrics"``.

    """
    metrics = FileTracker.Config()
    metrics.working_dir = working_dir
    tracker = TrackerList.Config()
    tracker.trackers = {"metrics": metrics}
    return tracker
