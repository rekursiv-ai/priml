"""Logging helpers: a context-manager timer and a colorized formatter.

Provides :class:`Timer` for scoped elapsed-time logging,
:class:`CustomFormatter` for rank-aware colorized log records, and
:func:`setup_logging` to wire the formatter onto the root logger.
"""

from __future__ import annotations

from types import TracebackType
from typing import ClassVar, Self, TextIO, cast, override

import datetime
import logging
import sys
import time

from configgle import Fig
from wrapt import lazy_import


dist = lazy_import("torch.distributed")


class Timer:
    """Context manager that logs the wall-clock time of its enclosed block."""

    __slots__: ClassVar[tuple[str, ...]] = (
        "description",
        "elapsed",
        "level",
        "logger",
        "start",
        "stop",
    )

    def __init__(
        self,
        description: str = "Timer",
        logger: logging.Logger | str = __name__,
        level: int = logging.INFO,
    ) -> None:
        self.description = description
        if isinstance(logger, str):
            logger = logging.getLogger(logger)
        self.logger: logging.Logger = logger
        self.level = level
        self.start: float = 0
        self.stop: float = 0
        self.elapsed: float = 0

    def __enter__(self) -> Self:
        self.start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        del exc_type, exc_val, exc_tb
        self.stop = time.perf_counter()
        self.elapsed = self.stop - self.start
        self.logger.log(
            self.level,
            f"{self.description}: {self.elapsed:.4f} seconds",
        )


class CustomFormatter(logging.Formatter):
    """Log formatter with per-level ANSI colors and distributed rank info."""

    FORMAT: ClassVar[str] = (
        "%(asctime)s | %(rank_info)s%(levelname)s | %(name)s%(classname)s | %(funcName)s:%(lineno)d | %(message)s"
    )
    DATEFMT: ClassVar[str] = "%y-%m-%d %H:%M:%S.%f"

    class LogColors:
        DEBUG: str = "\x1b[38;20m"
        INFO: str = "\x1b[34;20m"
        WARNING: str = "\x1b[33;20m"
        ERROR: str = "\x1b[31;20m"
        CRITICAL: str = "\x1b[31;1m"
        RESET: str = "\x1b[0m"

    LOG_LEVEL_COLOR_MAP: ClassVar[dict[int, str]] = {
        logging.DEBUG: LogColors.DEBUG + "DEBUG" + LogColors.RESET,
        logging.INFO: LogColors.INFO + "INFO" + LogColors.RESET,
        logging.WARNING: LogColors.WARNING + "WARNING" + LogColors.RESET,
        logging.ERROR: LogColors.ERROR + "ERROR" + LogColors.RESET,
        logging.CRITICAL: LogColors.CRITICAL + "CRITICAL" + LogColors.RESET,
    }

    class Config(Fig["CustomFormatter"]):
        fmt: str = "%(asctime)s | %(rank_info)s%(levelname)s | %(name)s%(classname)s | %(funcName)s:%(lineno)d | %(message)s"
        """Log message format string."""

        datefmt: str = "%y-%m-%d %H:%M:%S.%f"
        """Timestamp format string."""

    def __init__(self, config: Config) -> None:
        super().__init__(config.fmt, config.datefmt)
        self.datefmt = config.datefmt

    @override
    def format(self, record: logging.LogRecord) -> str:
        # Add rank information only if distributed is initialized.
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            record.rank_info = f"rank {rank}/{world_size} | "
        else:
            record.rank_info = ""

        # Class-name extraction is costly on this hot path; module name suffices.
        record.classname = ""

        original_levelname = record.levelname
        record.levelname = self.LOG_LEVEL_COLOR_MAP.get(
            record.levelno,
            original_levelname,
        )
        formatted_message = super().format(record)
        record.levelname = original_levelname
        return formatted_message

    @override
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.datetime.fromtimestamp(record.created)
        return dt.strftime(datefmt or self.datefmt or "")


class _ReplayBufferHandler(logging.Handler):
    """Retains emitted records so they can be replayed to a later stdout.

    W&B's console capture only hooks ``sys.stdout`` from ``wandb.init()``
    onward, so every record logged before init (hardware banner, experiment
    config, model parameter counts) is invisible in the W&B run. This handler
    keeps those records in memory; :func:`replay_buffered_logs`, called once
    after ``wandb.init()``, re-emits them through the now-wrapped stdout so the
    W&B console shows the full run log from the start. ``job.log`` already has
    them via the stream handler, so replay is purely for the W&B console.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    @override
    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _StdoutStreamHandler(logging.StreamHandler[TextIO]):
    """Loop-owned stdout stream handler that may be rebound after W&B init."""


def setup_logging(level: str = "INFO") -> None:
    """Setup logging with specified level.

    Args:
      level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    """
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root_logger.setLevel(numeric_level)
    handler = _StdoutStreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.setFormatter(CustomFormatter.Config().make())
    root_logger.addHandler(handler)

    # Buffer records so they can be replayed into the W&B console once
    # wandb.init() has hooked stdout (see replay_buffered_logs). The handler is
    # found via the root logger's handler list, so no module global is needed.
    buffer = _ReplayBufferHandler()
    buffer.setLevel(numeric_level)
    root_logger.addHandler(buffer)


def bind_logging_to_current_stdout() -> None:
    """Retarget Loop stdout logging handlers to the current ``sys.stdout``.

    W&B console capture wraps ``sys.stdout`` during ``wandb.init()``. Stream
    handlers constructed before that keep writing to the old stream unless they
    are explicitly rebound. Only Loop-owned handlers are updated; third-party
    handlers attached to the root logger are left untouched.
    """
    for handler in logging.getLogger().handlers:
        if isinstance(handler, _StdoutStreamHandler):
            handler.setStream(cast(TextIO, sys.stdout))


def replay_buffered_logs() -> None:
    """Replay pre-``wandb.init()`` log records to the current stdout.

    Re-emits everything buffered since :func:`setup_logging` through a fresh
    stream handler on the live ``sys.stdout`` (W&B-wrapped after init), so the
    W&B console captures the early log. Idempotent: detaches and drops the
    buffer after the first call. No-op when no buffer is installed.
    """
    root_logger = logging.getLogger()
    buffer = next(
        (h for h in root_logger.handlers if isinstance(h, _ReplayBufferHandler)),
        None,
    )
    if buffer is None:
        return
    replay = logging.StreamHandler(sys.stdout)
    replay.setFormatter(CustomFormatter.Config().make())
    for record in buffer.records:
        replay.emit(record)
    replay.flush()
    root_logger.removeHandler(buffer)
