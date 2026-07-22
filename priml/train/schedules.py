"""LR schedule functions: step → multiplier in [0, 1]."""

from __future__ import annotations

from collections.abc import Callable

import math


_Schedule = Callable[[int, int], float]
"""A schedule: ``(step, total_steps) -> multiplier``."""


def multiply_schedules(*schedules: _Schedule) -> _Schedule:
    """Combine schedules by multiplying their multipliers.

    Bind per-schedule keyword args with ``functools.partial`` first, e.g.::

        schedule = multiply_schedules(
            functools.partial(warmup, warmup_fraction=0.05),
            functools.partial(polynomial, power=2.0),
        )

    Args:
      *schedules: ``(step, total_steps) -> multiplier`` callables.

    Returns:
      schedule: A schedule returning the product of all multipliers (1.0
        when no schedules are given).

    """

    def combined(step: int, total_steps: int) -> float:
        result = 1.0
        for schedule in schedules:
            result *= schedule(step, total_steps)
        return result

    return combined


def polynomial(step: int, total_steps: int, *, power: float = 1.0) -> float:
    """Polynomial decay: (1 - t)^power."""
    t = step / max(1, total_steps)
    return max(0.0, (1 - t) ** power)


def cosine(step: int, total_steps: int) -> float:
    """Cosine annealing: 0.5 * (1 + cos(pi * t))."""
    t = step / max(1, total_steps)
    return 0.5 * (1 + math.cos(math.pi * t))


def trapezoidal(
    step: int,
    total_steps: int,
    *,
    warmup_steps: int = 0,
    decay_fraction: float = 0.3,
    cooldown_cosine: bool = False,
) -> float:
    """Trapezoidal: linear warmup → flat → decay to 0.

    Args:
      step: Current step.
      total_steps: Total training steps.
      warmup_steps: Steps for linear warmup.
      decay_fraction: Fraction of total steps spent decaying (from end).
      cooldown_cosine: If True, use cosine decay; otherwise linear.

    """
    if warmup_steps > 0 and step < warmup_steps:
        return step / max(1, warmup_steps)
    decay_start = int((1 - decay_fraction) * total_steps)
    if step < decay_start:
        return 1.0
    t = (step - decay_start) / max(1, total_steps - decay_start)
    t = min(1.0, t)
    if cooldown_cosine:
        return 0.5 * (1 + math.cos(math.pi * t))
    return 1.0 - t


def warmup(step: int, total_steps: int, *, warmup_fraction: float) -> float:
    """Linear warmup multiplier.

    Returns 1.0 after warmup is complete. Combine with a decay schedule via
    :func:`multiply_schedules` (binding kwargs with ``functools.partial``).
    """
    if warmup_fraction <= 0:
        return 1.0
    t = step / max(1, total_steps)
    if t < warmup_fraction:
        return t / warmup_fraction
    return 1.0
