"""Tests for learning-rate schedules.

Every curve here is a function of progress alone, so each case pins a property
a reader could otherwise only get by running a job: where the rate starts,
where it lands, and what it does to the run in between.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import itertools
import math

import pytest

from priml.math.schedules import (
    Schedule,
    constant,
    cosine,
    cosine_restarts,
    exponential,
    linear,
    multiply_schedules,
    one_cycle,
    polynomial,
    staircase,
    trapezoidal,
    warmup,
)


DECAYS: list[tuple[str, Schedule[float]]] = [
    ("linear", linear),
    ("polynomial", polynomial),
    ("cosine", cosine),
    ("exponential", partial(exponential, decay=1e-3)),
    ("trapezoidal", trapezoidal),
    ("one_cycle", one_cycle),
]

EVERY: list[tuple[str, Schedule[float]]] = [
    *DECAYS,
    ("constant", constant),
    ("staircase", staircase),
    ("warmup", partial(warmup, fraction=0.1)),
    ("cosine_restarts", cosine_restarts),
]

IDS = [name for name, _ in EVERY]


@pytest.mark.parametrize(("name", "schedule"), EVERY, ids=IDS)
def test_no_schedule_leaves_its_own_range(
    name: str,
    schedule: Schedule[float],
) -> None:
    """A multiplier outside ``[0, 1]`` would raise the rate above its own peak.

    Every curve here scales an initial rate, so one is the ceiling by
    construction -- a schedule that exceeded it would be tuning the rate rather
    than scheduling it.
    """
    del name
    for index in range(101):
        value = schedule(index / 100)
        assert 0.0 <= value <= 1.0


@pytest.mark.parametrize(("name", "schedule"), EVERY, ids=IDS)
def test_progress_outside_the_unit_interval_is_clamped(
    name: str,
    schedule: Schedule[float],
) -> None:
    """A budget is checked between steps, so the last one lands past it.

    Unclamped, a polynomial would raise a negative base to a fractional power
    and a cosine would climb back toward its peak -- the run's final steps
    training at a rate the recipe never specified.
    """
    del name
    assert schedule(1.5) == schedule(1.0)
    assert schedule(-0.5) == schedule(0.0)


@pytest.mark.parametrize(
    ("name", "schedule"),
    [(n, s) for n, s in DECAYS if n != "exponential"],
    ids=[n for n, _ in DECAYS if n != "exponential"],
)
def test_every_decay_lands_at_zero(
    name: str,
    schedule: Schedule[float],
) -> None:
    """The last weights should be an average over a low-noise tail.

    A schedule that stopped short leaves the run ending mid-oscillation, which
    is what annealing to zero exists to avoid.

    ``exponential`` is excluded because it cannot reach zero -- a geometric
    decay approaches it -- which is why its parameter is the multiplier it
    lands ON rather than a rate it decays at.
    """
    del name
    assert schedule(1.0) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("name", "schedule"),
    [(n, s) for n, s in DECAYS if n != "one_cycle"],
    ids=[n for n, _ in DECAYS if n != "one_cycle"],
)
def test_every_decay_starts_at_the_full_rate(
    name: str,
    schedule: Schedule[float],
) -> None:
    """Decay scales the rate the recipe was tuned at, so it starts there.

    ``one_cycle`` is excluded because it deliberately does not: it ramps up to
    the full rate rather than starting at it.
    """
    del name
    assert schedule(0.0) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("name", "schedule"),
    [(n, s) for n, s in DECAYS if n != "one_cycle"],
    ids=[n for n, _ in DECAYS if n != "one_cycle"],
)
def test_every_decay_is_monotone(
    name: str,
    schedule: Schedule[float],
) -> None:
    """A decay that rose again would re-heat a basin the run had settled into."""
    del name
    values = [schedule(index / 200) for index in range(201)]
    for earlier, later in itertools.pairwise(values):
        assert later <= earlier + 1e-12


def test_constant_never_moves() -> None:
    assert {constant(index / 10) for index in range(11)} == {1.0}


def test_linear_halves_at_the_midpoint() -> None:
    assert linear(0.5) == pytest.approx(0.5)


def test_linear_lands_on_its_floor() -> None:
    """A nonzero floor is what keeps a run learning through its tail."""
    assert linear(1.0, final=0.1) == pytest.approx(0.1)
    assert linear(0.5, final=0.1) == pytest.approx(0.55)


def test_polynomial_power_shifts_where_the_rate_is_spent() -> None:
    """Higher powers hold the rate up, then drop it late.

    The knob's whole purpose: at the midpoint a quadratic still has a quarter
    of the rate where a linear has half.
    """
    assert polynomial(0.5, power=1.0) == pytest.approx(0.5)
    assert polynomial(0.5, power=2.0) == pytest.approx(0.25)
    assert polynomial(0.5, power=0.5) == pytest.approx(math.sqrt(0.5))


def test_cosine_is_flat_at_both_ends() -> None:
    """The shape's reason for being: it neither starts nor stops abruptly."""
    assert cosine(0.01) > 0.999
    assert cosine(0.99) < 0.001
    assert cosine(0.5) == pytest.approx(0.5)


def test_exponential_reaches_its_stated_decay() -> None:
    """The parameter is the FINAL multiplier, not a per-step rate.

    A per-step ``gamma`` would mean different things on two runs of different
    length; this one means the same thing on both.
    """
    assert exponential(1.0, decay=0.01) == pytest.approx(0.01)
    assert exponential(0.5, decay=0.01) == pytest.approx(0.1)


def test_exponential_rejects_a_decay_it_cannot_reach() -> None:
    """No geometric decay reaches zero or a negative multiplier."""
    with pytest.raises(ValueError, match="decay must be positive"):
        exponential(0.5, decay=0.0)


def test_staircase_drops_the_stated_number_of_times() -> None:
    """The count is what survives a change of horizon.

    A step-indexed decay would drop a different number of times on a longer
    run, silently changing the recipe.
    """
    assert staircase(0.0, drops=3, gamma=0.1) == pytest.approx(1.0)
    assert staircase(0.3, drops=3, gamma=0.1) == pytest.approx(0.1)
    assert staircase(0.6, drops=3, gamma=0.1) == pytest.approx(0.01)
    assert staircase(0.8, drops=3, gamma=0.1) == pytest.approx(0.001)
    assert staircase(1.0, drops=3, gamma=0.1) == pytest.approx(0.001)


def test_staircase_without_drops_is_flat() -> None:
    assert staircase(0.9, drops=0) == 1.0


def test_staircase_rejects_a_negative_count() -> None:
    with pytest.raises(ValueError, match="drops must be nonnegative"):
        staircase(0.5, drops=-1)


def test_warmup_ramps_then_gets_out_of_the_way() -> None:
    """It returns to 1.0 because it is a FACTOR, not a schedule.

    Left below 1.0 afterwards it would suppress whatever decay it multiplies
    for the rest of the run.
    """
    assert warmup(0.0, fraction=0.1) == 0.0
    assert warmup(0.05, fraction=0.1) == pytest.approx(0.5)
    assert warmup(0.1, fraction=0.1) == pytest.approx(1.0)
    assert warmup(0.9, fraction=0.1) == 1.0


def test_warmup_disabled_is_the_identity() -> None:
    """Zero must mean no ramp, not a division by zero."""
    assert warmup(0.0, fraction=0.0) == 1.0


def test_trapezoidal_holds_then_decays() -> None:
    assert trapezoidal(0.0, flat=0.5) == 1.0
    assert trapezoidal(0.49, flat=0.5) == 1.0
    assert trapezoidal(0.75, flat=0.5) == pytest.approx(0.5)
    assert trapezoidal(1.0, flat=0.5) == pytest.approx(0.0)


def test_trapezoidal_cosine_cooldown_shares_its_endpoints() -> None:
    """The shape of the tail is a choice; where it starts and ends is not."""
    for flat in (0.0, 0.5, 0.9):
        linear_tail = partial(trapezoidal, flat=flat)
        cosine_tail = partial(trapezoidal, flat=flat, cooldown_cosine=True)
        assert cosine_tail(flat) == pytest.approx(linear_tail(flat))
        assert cosine_tail(1.0) == pytest.approx(linear_tail(1.0), abs=1e-9)


def test_trapezoidal_rejects_a_run_with_no_tail() -> None:
    """``flat=1.0`` would divide by zero and leave the rate never annealing."""
    with pytest.raises(ValueError, match=r"flat must lie in \[0, 1\)"):
        trapezoidal(0.5, flat=1.0)


def test_one_cycle_peaks_where_its_ramp_ends() -> None:
    """Both legs meet at exactly the full rate, so the peak is continuous."""
    assert one_cycle(0.0, warmup_fraction=0.3) == pytest.approx(0.04)
    assert one_cycle(0.3, warmup_fraction=0.3) == pytest.approx(1.0)
    assert one_cycle(1.0, warmup_fraction=0.3) == pytest.approx(0.0)


def test_one_cycle_rises_then_falls() -> None:
    values = [one_cycle(index / 100, warmup_fraction=0.3) for index in range(101)]
    peak = values.index(max(values))
    assert values[:peak] == sorted(values[:peak])
    assert values[peak:] == sorted(values[peak:], reverse=True)


def test_one_cycle_rejects_an_all_ramp_run() -> None:
    with pytest.raises(ValueError, match=r"warmup_fraction must lie in \[0, 1\)"):
        one_cycle(0.5, warmup_fraction=1.0)


def test_cosine_restarts_returns_to_the_full_rate() -> None:
    """The sawtooth IS the mechanism: each restart re-heats the basin."""
    assert cosine_restarts(0.0, cycles=3) == pytest.approx(1.0)
    assert cosine_restarts(1 / 3, cycles=3) == pytest.approx(1.0)
    assert cosine_restarts(2 / 3, cycles=3) == pytest.approx(1.0)
    assert cosine_restarts(1 / 6, cycles=3) == pytest.approx(0.5)


def test_cosine_restarts_ends_annealed_not_restarted() -> None:
    """The last point lands on a boundary, which reads as a restart.

    Left uncorrected the run would finish at its FULL rate -- the opposite of
    what every other decay here guarantees.
    """
    assert cosine_restarts(1.0, cycles=3) == pytest.approx(0.0)


def test_cosine_restarts_rejects_a_run_with_no_cycle() -> None:
    with pytest.raises(ValueError, match="cycles must be positive"):
        cosine_restarts(0.5, cycles=0)


def test_multiply_schedules_composes_warmup_with_decay() -> None:
    """The pairing the module exists to make readable.

    Each half stays a curve a reader can name, and neither has to know the
    other exists.
    """
    schedule = multiply_schedules(
        partial(warmup, fraction=0.1),
        partial(polynomial, power=1.0),
    )
    assert schedule(0.05) == pytest.approx(0.5 * polynomial(0.05))
    assert schedule(0.5) == pytest.approx(polynomial(0.5))


def test_multiply_schedules_with_nothing_is_the_identity() -> None:
    empty: Callable[[float], float] = multiply_schedules()
    assert empty(0.7) == 1.0


def test_multiply_schedules_carries_a_phased_progress() -> None:
    """The combinator is generic over what progress IS.

    A phased run reports ``(phase, fraction)``; composing two such schedules
    must not require either to be a schedule of a bare float.
    """

    def ramp(progress: tuple[str, float]) -> float:
        return progress[1]

    def half(progress: tuple[str, float]) -> float:
        del progress
        return 0.5

    assert multiply_schedules(ramp, half)(("warmup", 0.4)) == pytest.approx(0.2)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
