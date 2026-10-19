"""Indexed absolute rates for recipes whose recurrence rounding is significant."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import math

from configgle import Fig


@runtime_checkable
class AbsoluteRateSchedule(Protocol):
    """Compute a group's complete rate from its initial rate and integer clock."""

    def learning_rate(self, initial_lr: float, *, step: int) -> float:
        """Return a complete rate at the caller's scheduling index."""
        ...


class RecursiveCosine:
    """Index the binary64 cosine recurrence without owning an optimizer clock.

    Cached prefixes only save arithmetic: querying an earlier index, querying
    twice, or constructing a fresh schedule produces the same rate. A caller
    checkpoints its integer clock, not this cache. Unlike the progress cosine,
    this preserves the rounding of successive ratio/multiply/add updates.
    """

    class Config(Fig["RecursiveCosine"]):
        T_max: int = 1
        """Number of updates from the initial rate to the minimum."""

        eta_min: float = 0.0
        """Absolute minimum learning rate shared by all parameter groups."""

    def __init__(self, config: Config) -> None:
        if config.T_max <= 0:
            raise ValueError("T_max must be positive.")
        if not math.isfinite(config.eta_min):
            raise ValueError("eta_min must be finite.")
        self.T_max = config.T_max
        self.eta_min = config.eta_min
        self._rates: dict[float, list[float]] = {}

    def learning_rate(self, initial_lr: float, *, step: int) -> float:
        """Return the recurrence at a caller-owned update or epoch index.

        Args:
          initial_lr: This parameter group's rate before scheduling.
          step: Nonnegative number of completed scheduling intervals.

        Returns:
          rate: Binary64 rate, including the ascending half after ``T_max``.

        """
        if step < 0:
            raise ValueError("step must be nonnegative.")
        rates = self._rates.setdefault(initial_lr, [initial_lr])
        while len(rates) <= step:
            index = len(rates)
            previous = rates[-1]
            if (index - 1) % (2 * self.T_max) == self.T_max:
                rate = (
                    previous
                    + (initial_lr - self.eta_min)
                    * (1 - math.cos(math.pi / self.T_max))
                    / 2
                )
            else:
                rate = (1 + math.cos(math.pi * index / self.T_max)) / (
                    1 + math.cos(math.pi * (index - 1) / self.T_max)
                ) * (previous - self.eta_min) + self.eta_min
            rates.append(rate)
        return rates[step]
