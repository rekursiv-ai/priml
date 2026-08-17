"""Learning-rate schedules: progress in ``[0, 1]``, multiplier out.

A schedule holds no state, reads no clock, and touches no optimizer, so what
"how far" MEANS belongs to the caller -- which is what lets one set of curves
serve a run budgeted in steps, in seconds, or in tokens, and what makes them
exactly reproducible on resume where a stateful scheduler's counter can
desync.

Combine curves with :func:`multiply_schedules`; warmup times decay is the
usual pairing. Nothing here computes progress: see
:class:`~priml.train.train_step.TrainStep`, which owns the budget and
writes the multiplier into ``param_groups``.
"""

from __future__ import annotations

from collections.abc import Callable

import math


__all__ = [
    "Schedule",
    "constant",
    "cosine",
    "cosine_restarts",
    "exponential",
    "linear",
    "multiply_schedules",
    "one_cycle",
    "polynomial",
    "staircase",
    "trapezoidal",
    "warmup",
]


type Schedule[ProgressT] = Callable[[ProgressT], float]
"""Maps a run's progress to a learning-rate multiplier.

Generic over what progress IS so a phased run can report ``(phase, fraction)``
and be scheduled by something that dispatches on the phase, while an ordinary
run reports a bare fraction. Every curve in this module is a
``Schedule[float]``; a phased schedule is a callable a caller writes, holding
one of these per phase.
"""


def multiply_schedules[ProgressT](
    *schedules: Schedule[ProgressT],
) -> Schedule[ProgressT]:
    """Combine schedules by multiplying their multipliers.

    The composition operator for this module: warmup and decay are separate
    curves precisely so a reader can see each one, and multiplying is what
    joins them without either having to know about the other.

    Bind per-schedule keyword arguments with :func:`functools.partial` first::

        schedule = multiply_schedules(
            partial(warmup, fraction=0.05),
            partial(polynomial, power=2.0),
        )

    Args:
      *schedules: Curves to multiply.

    Returns:
      schedule: Their product; the constant 1.0 when given none.

    """

    def combined(progress: ProgressT) -> float:
        result = 1.0
        for schedule in schedules:
            result *= schedule(progress)
        return result

    return combined


def constant(progress: float) -> float:
    """Hold the rate flat for the whole run.

    Args:
      progress: Fraction of the budget spent.

    Returns:
      multiplier: Always 1.0.

    """
    del progress
    return 1.0


def linear(progress: float, *, final: float = 0.0) -> float:
    """Decay linearly from the full rate to ``final``.

    Args:
      progress: Fraction of the budget spent.
      final: Multiplier at the end of the run.

    Returns:
      multiplier: The rate's share of its initial value.

    """
    return 1.0 + (final - 1.0) * _clamped(progress)


def polynomial(progress: float, *, power: float = 1.0) -> float:
    """Decay as ``(1 - progress) ** power``.

    Args:
      progress: Fraction of the budget spent.
      power: Curve shape; 1 is linear, higher spends longer at a high rate.

    Returns:
      multiplier: The rate's share of its initial value.

    """
    return math.pow(1.0 - _clamped(progress), power)


def cosine(progress: float, *, final: float = 0.0) -> float:
    """Anneal along a half cosine from the full rate to ``final``.

    Leaves the rate high early and lands flat, so the last steps average over
    a low-noise tail rather than stopping mid-oscillation.

    Args:
      progress: Fraction of the budget spent.
      final: Multiplier at the end of the run.

    Returns:
      multiplier: The rate's share of its initial value.

    """
    shape = 0.5 * (1.0 + math.cos(math.pi * _clamped(progress)))
    return final + (1.0 - final) * shape


def exponential(progress: float, *, decay: float) -> float:
    """Decay geometrically, reaching ``decay`` at the end of the run.

    Stated as the FINAL multiplier rather than a per-step rate: a per-step
    ``gamma`` is only meaningful beside a step count, which a schedule of
    progress does not have -- and the same ``gamma`` would then mean different
    things on two runs of different length.

    Args:
      progress: Fraction of the budget spent.
      decay: Multiplier at the end of the run; must be positive.

    Returns:
      multiplier: The rate's share of its initial value.

    Raises:
      ValueError: ``decay`` is not positive, which no geometric decay reaches.

    """
    if decay <= 0:
        raise ValueError(f"decay must be positive; got {decay}.")
    return math.pow(decay, _clamped(progress))


def staircase(progress: float, *, drops: int = 3, gamma: float = 0.1) -> float:
    """Hold flat, then drop by ``gamma`` at evenly spaced points.

    The progress-native form of a step decay: ``drops`` says how many times the
    rate falls, so the schedule keeps its shape when the horizon changes --
    where a step-indexed one would silently drop a different number of times.

    Args:
      progress: Fraction of the budget spent.
      drops: Number of drops across the run.
      gamma: Factor applied at each drop.

    Returns:
      multiplier: The rate's share of its initial value.

    Raises:
      ValueError: ``drops`` is negative.

    """
    if drops < 0:
        raise ValueError(f"drops must be nonnegative; got {drops}.")
    if drops == 0:
        return 1.0
    # ``min`` rather than a bare floor: progress reaches exactly 1.0 at the
    # end, which would otherwise count one drop more than the run has.
    taken = min(int(_clamped(progress) * (drops + 1)), drops)
    return math.pow(gamma, taken)


def warmup(progress: float, *, fraction: float) -> float:
    """Ramp the rate linearly from zero over the run's opening.

    Returns 1.0 once the ramp is done, so this is a FACTOR to multiply a decay
    by rather than a schedule in itself -- see :func:`multiply_schedules`.

    Args:
      progress: Fraction of the budget spent.
      fraction: Share of the run spent ramping; 0 disables the ramp.

    Returns:
      multiplier: The rate's share of its initial value.

    """
    if fraction <= 0:
        return 1.0
    return min(1.0, _clamped(progress) / fraction)


def trapezoidal(
    progress: float,
    *,
    flat: float = 0.5,
    final: float = 0.0,
    cooldown_cosine: bool = False,
) -> float:
    """Hold the rate flat, then decay it over the run's tail.

    Spends most of the budget at the full rate while still landing at
    ``final``, which is what makes the last weights an average over a
    low-noise tail rather than a snapshot mid-oscillation.

    Args:
      progress: Fraction of the budget spent.
      flat: Share of the run held at the full rate.
      final: Multiplier at the end of the run.
      cooldown_cosine: Decay along a half cosine rather than linearly.

    Returns:
      multiplier: The rate's share of its initial value.

    Raises:
      ValueError: ``flat`` is outside ``[0, 1)``, leaving no tail to decay over.

    """
    if flat < 0.0 or flat >= 1.0:
        raise ValueError(f"flat must lie in [0, 1); got {flat}.")
    spent = _clamped(progress)
    if spent < flat:
        return 1.0
    tail = (spent - flat) / (1.0 - flat)
    return cosine(tail, final=final) if cooldown_cosine else linear(tail, final=final)


def one_cycle(
    progress: float,
    *,
    warmup_fraction: float = 0.3,
    initial: float = 0.04,
    final: float = 0.0,
) -> float:
    """Ramp up to the full rate, then anneal past it toward ``final``.

    Both legs are cosine, which is what distinguishes this from a warmup
    multiplied by a decay: the ramp is a curve in its own right rather than a
    correction applied to one.

    Args:
      progress: Fraction of the budget spent.
      warmup_fraction: Share of the run spent ramping up.
      initial: Multiplier the ramp starts from.
      final: Multiplier the anneal ends at.

    Returns:
      multiplier: The rate's share of its peak value.

    Raises:
      ValueError: ``warmup_fraction`` is outside ``[0, 1)``.

    """
    if warmup_fraction < 0.0 or warmup_fraction >= 1.0:
        raise ValueError(
            f"warmup_fraction must lie in [0, 1); got {warmup_fraction}.",
        )
    spent = _clamped(progress)
    if spent < warmup_fraction:
        rising = spent / warmup_fraction if warmup_fraction else 1.0
        # The rising leg is the falling cosine read backwards, so both legs
        # meet at exactly 1.0 and the peak has no discontinuity.
        return initial + (1.0 - initial) * (1.0 - cosine(rising))
    tail = (spent - warmup_fraction) / (1.0 - warmup_fraction)
    return cosine(tail, final=final)


def cosine_restarts(
    progress: float,
    *,
    cycles: int = 3,
    final: float = 0.0,
) -> float:
    """Anneal along a cosine, restarting at the full rate ``cycles`` times.

    Each restart is a jump back to 1.0, so this is the one curve here that is
    not monotone -- progress still is, and the sawtooth is the point.

    Args:
      progress: Fraction of the budget spent.
      cycles: Equal-length cycles across the run.
      final: Multiplier at the end of each cycle.

    Returns:
      multiplier: The rate's share of its initial value.

    Raises:
      ValueError: ``cycles`` is not positive.

    """
    if cycles <= 0:
        raise ValueError(f"cycles must be positive; got {cycles}.")
    # The final point lands exactly on a boundary; ``% 1.0`` would read it as
    # the START of a cycle that does not exist and return the full rate.
    within = math.fmod(_clamped(progress) * cycles, 1.0)
    if within == 0.0 and progress >= 1.0:
        return final
    return cosine(within, final=final)


def _clamped(progress: float) -> float:
    """Clamp progress to ``[0, 1]``.

    A caller's clock can overshoot -- a budget is checked between steps, so the
    last one lands past it -- and every curve here would then run off its own
    domain: a polynomial would raise a negative base to a fractional power, and
    a cosine would climb back up.
    """
    return min(1.0, max(0.0, progress))
