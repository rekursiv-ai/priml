"""Outer wall accounting around a caller-owned preparation, fit and publication."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import FrameType
from typing import TYPE_CHECKING, Self

import json
import logging
import math
import os
import signal
import time

from configgle import Fig
from treekle import to_plain

from priml.paths import validated_output_path


if TYPE_CHECKING:
    from types import TracebackType


logger = logging.getLogger(__name__)


class WallBudget:
    """Measure from construction; seal a receipt when the context exits.

    Construct before preparing data or building the workload. The caller owns
    preparation, training, evaluation and export inside ``with budget:``. Add
    local provenance to ``metadata``; it cannot override receipt status or time.
    Use ``with budget.interruptible():`` inside that context for a POSIX main
    thread deadline. Receipt sealing occurs after the alarm scope is restored.

    Receipt elapsed time is sampled before serialization; admission is checked
    again after the file and directory are durable. A late seal is replaced by
    a failure receipt and raises TimeoutError. Filesystem errors cannot replace
    an existing workload exception: they are attached to it as notes instead.
    """

    class Config(Fig["WallBudget"]):
        """Declare a wall allowance without constructing the workload."""

        seconds: float = math.inf
        """Positive outer allowance, including preparation and receipt publication."""

        reserve_evaluation_sec: float = 0
        """Finite nonnegative allowance withheld by training_seconds()."""

        clock: Callable[[], float] = time.perf_counter
        """Monotonic clock, sampled first when the runtime is constructed."""

        receipt_path: Path | None = None
        """Literal optional JSON destination; no template expansion is performed."""

    def __init__(self, config: Config) -> None:
        if math.isnan(config.seconds) or config.seconds <= 0:
            raise ValueError("Wall budget seconds must be positive or infinity.")
        if (
            not math.isfinite(config.reserve_evaluation_sec)
            or config.reserve_evaluation_sec < 0
            or config.reserve_evaluation_sec >= config.seconds
        ):
            raise ValueError("Evaluation reserve must be finite and within the budget.")
        self.seconds = config.seconds
        self.started = config.clock()
        self.clock = config.clock
        self.reserve_evaluation_sec = config.reserve_evaluation_sec
        self.receipt_path = config.receipt_path
        self.metadata: dict[str, object] = {}
        self._entered = False

    def __enter__(self) -> Self:
        """Enter this single-use accounting context without resetting its clock."""
        if self._entered:
            raise RuntimeError("A WallBudget context can only be entered once.")
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Seal success or failure without replacing a workload exception."""
        del exc_type, traceback
        if error is not None:
            self._fail(error)
            return
        try:
            self.check()
            self._publish(None)
            self.check()
        except BaseException as seal_error:
            self._fail(seal_error)
            raise

    @property
    def elapsed(self) -> float:
        """Return seconds since construction, including work before context entry."""
        return self.clock() - self.started

    @property
    def remaining(self) -> float:
        """Return the unclamped outer allowance remaining."""
        return self.seconds - self.elapsed

    def training_seconds(self) -> float:
        """Return positive training time after withholding evaluation reserve."""
        remaining = self.remaining - self.reserve_evaluation_sec
        if remaining <= 0:
            raise TimeoutError("No training time remains after evaluation reserve.")
        return remaining

    def check(self) -> None:
        """Raise TimeoutError when the outer allowance has been exceeded."""
        if self.elapsed > self.seconds:
            raise TimeoutError("Outer wall budget exceeded.")

    @contextmanager
    def interruptible(self) -> Generator[None]:
        """Interrupt at the remaining deadline, sharing the caller's real alarm.

        The caller's earlier alarm is dispatched with its original frame and
        interval. Both deadlines remain armed while its handler runs. On exit,
        restore its handler and remaining timer, charging time spent here.
        This requires POSIX SIGALRM and the main thread, as signal.signal does.

        Yields:
          scope: Work covered by the remaining outer deadline.

        """
        remaining = self.remaining
        if math.isinf(remaining):
            yield
            return
        if remaining <= 0:
            raise TimeoutError("Outer wall budget exceeded.")
        before = time.perf_counter()
        previous = signal.getsignal(signal.SIGALRM)
        timer, interval = signal.getitimer(signal.ITIMER_REAL)
        alarm = _Alarm(
            deadline=before + remaining,
            caller_due=before + timer if timer > 0 else None,
            interval=interval,
            previous=previous,
        )
        signal.signal(signal.SIGALRM, alarm.expired)
        try:
            alarm.arm()
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            signal.setitimer(
                signal.ITIMER_REAL,
                max(1e-6, alarm.caller_due - time.perf_counter())
                if alarm.caller_due is not None
                else 0,
                interval,
            )

    def _publish(self, error: BaseException | None) -> None:
        if self.receipt_path is None:
            return
        path = validated_output_path(self.receipt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, object] = {
            "status": "complete" if error is None else "failed",
            "budget_seconds": self.seconds,
            "elapsed_seconds": self.elapsed,
            "metadata": self.metadata,
        }
        if error is not None:
            record["error"] = f"{type(error).__name__}: {error}"
        with NamedTemporaryFile(
            prefix="budget-",
            mode="w",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            try:
                _ = stream.write(json.dumps(to_plain(record), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                _ = temporary.replace(path)
                directory = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                temporary.unlink(missing_ok=True)

    def _fail(self, error: BaseException) -> None:
        try:
            self._publish(error)
        except BaseException as publication_error:
            logger.exception("Failure receipt could not be published.")
            error.add_note(f"Receipt publication failed: {publication_error}")
            # A failed failure-rewrite must not leave a previous success behind.
            try:
                if self.receipt_path is not None:
                    validated_output_path(self.receipt_path).unlink(missing_ok=True)
            except BaseException as cleanup_error:
                logger.exception("Failure receipt could not be invalidated.")
                error.add_note(f"Receipt invalidation failed: {cleanup_error}")


@dataclass(slots=True, kw_only=True)
class _Alarm:
    """Share the real timer without hiding an earlier caller deadline."""

    deadline: float
    caller_due: float | None
    interval: float
    previous: Callable[[int, FrameType | None], object] | int | None

    def arm(self) -> None:
        target = (
            min(self.deadline, self.caller_due)
            if self.caller_due is not None
            else self.deadline
        )
        signal.setitimer(signal.ITIMER_REAL, max(1e-6, target - time.perf_counter()))

    def expired(self, signum: int, frame: FrameType | None) -> None:
        if self.caller_due is None or self.deadline <= self.caller_due:
            raise TimeoutError("Outer wall budget exceeded.")
        self.caller_due = self.caller_due + self.interval if self.interval else None
        self.arm()
        if callable(self.previous):
            self.previous(signum, frame)
        elif self.previous == signal.SIG_DFL:
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            signal.raise_signal(signal.SIGALRM)
