"""A counted, timed activity whose totals survive a resume.

Separate from :class:`priml.logger.Timer`, which logs how long a block
took and remembers nothing: this one is CHECKPOINT state. A run's schedules and
its stop condition read it, so it has to mean the same thing after a resume as
it did before one -- which is what the global/local split below is for.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import time


__all__ = [
    "CheckpointableStepTimer",
]


@dataclass(slots=True, kw_only=True)
class CheckpointableStepTimer:
    """Times and counts one activity, and survives a resume.

    Used as the context itself (``with self.timer_step:``), and not
    re-entrant: a nested enter would charge and count the inner block twice.

    Seconds are HOST-side wall time with no accelerator sync, since a sync per
    forward would serialize the pipeline this measures. A single reading is
    therefore launch time; over many calls the queue saturates and the total
    converges on device time.
    """

    global_count: int = 0
    """Times run across the whole run, resumes included; restored."""

    local_count: int = 0
    """Times run since this process started; deliberately not restored.

    What separates "this job has done 40 steps" from "this run has done
    40,000" -- the distinction a warmup exclusion or a first-step branch needs
    after a resume, and one the global count alone cannot make."""

    global_sec: float = 0.0
    """Seconds spent inside across the whole run; restored."""

    local_sec: float = 0.0
    """Seconds spent inside since this process started; not restored.

    Accumulated from a monotonic clock whose origin is this process's own, so
    a value carried in from another one measures nothing."""

    _started: float | None = field(default=None, init=False, repr=False)
    """When the open block began; None between blocks.

    Not a constructor argument and not checkpointed: it is a reading from THIS
    process's monotonic clock, so a value restored from another one measures
    nothing. A timer is therefore always saved closed."""

    def __enter__(self) -> Self:
        """Start timing one run of the block.

        Returns:
          timer: This timer, so ``with`` may bind it.

        Raises:
          RuntimeError: A block is already open. Nesting would charge the inner
            one twice, so it is refused rather than silently miscounted.

        """
        if self._started is not None:
            raise RuntimeError(
                "this timer is already running; a nested block would be "
                "counted twice. Use a separate timer for the inner work.",
            )
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Charge the block its time and count it, raised or not.

        A call that raised still happened, and a record that hid it would
        misreport the very step being debugged.
        """
        del exc_info
        assert self._started is not None
        elapsed = time.perf_counter() - self._started
        self._started = None
        self.global_count += 1
        self.local_count += 1
        self.global_sec += elapsed
        self.local_sec += elapsed

    def state_dict(self) -> dict[str, Any]:
        """Return the global totals; the local ones belong to this process."""
        return {"global_count": self.global_count, "global_sec": self.global_sec}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore the global totals and zero the local ones."""
        self.global_count = int(state_dict["global_count"])
        self.global_sec = float(state_dict["global_sec"])
        self.local_count = 0
        self.local_sec = 0.0
