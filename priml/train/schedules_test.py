"""Tests for LR schedules."""

from __future__ import annotations

import functools
import math

from priml.train.schedules import (
    cosine,
    multiply_schedules,
    polynomial,
    trapezoidal,
    warmup,
)


def test_polynomial_linear_decay():
    assert polynomial(0, 100, power=1.0) == 1.0
    assert polynomial(100, 100, power=1.0) == 0.0
    assert polynomial(50, 100, power=1.0) == 0.5


def test_polynomial_quadratic():
    assert polynomial(50, 100, power=2.0) == 0.25


def test_polynomial_clamps_negative():
    assert polynomial(200, 100) == 0.0


def test_cosine_endpoints():
    assert cosine(0, 100) == 1.0
    assert math.isclose(cosine(100, 100), 0.0, abs_tol=1e-9)


def test_cosine_midpoint():
    assert math.isclose(cosine(50, 100), 0.5, abs_tol=1e-9)


def test_trapezoidal_warmup():
    assert trapezoidal(0, 100, warmup_steps=10) == 0.0
    assert math.isclose(trapezoidal(5, 100, warmup_steps=10), 0.5)
    assert trapezoidal(10, 100, warmup_steps=10) == 1.0


def test_trapezoidal_flat():
    assert trapezoidal(50, 100, decay_fraction=0.3) == 1.0


def test_trapezoidal_decay():
    v = trapezoidal(85, 100, decay_fraction=0.3)
    assert 0.0 < v < 1.0
    assert trapezoidal(100, 100, decay_fraction=0.3) == 0.0


def test_trapezoidal_no_warmup():
    assert trapezoidal(0, 100) == 1.0


def test_warmup_linear():
    assert warmup(0, 100, warmup_fraction=0.1) == 0.0
    assert math.isclose(warmup(5, 100, warmup_fraction=0.1), 0.5)
    assert warmup(10, 100, warmup_fraction=0.1) == 1.0
    assert warmup(50, 100, warmup_fraction=0.1) == 1.0


def test_warmup_disabled():
    assert warmup(0, 100, warmup_fraction=0.0) == 1.0


def test_multiply_schedules_multiplies_multipliers():
    """T-054: multiply_schedules() multiplies multipliers (warmup * decay)."""
    warm = functools.partial(warmup, warmup_fraction=0.1)
    decay = functools.partial(polynomial, power=1.0)
    schedule = multiply_schedules(warm, decay)

    # At step 5 (mid-warmup): warmup=0.5, decay=(1-0.05)=0.95 -> 0.475.
    assert math.isclose(schedule(5, 100), 0.5 * polynomial(5, 100))
    # Post-warmup the warmup factor is 1.0, so it equals decay alone.
    assert math.isclose(schedule(50, 100), polynomial(50, 100))


def test_multiply_schedules_empty_is_identity():
    """multiply_schedules() with no schedules returns the constant-1 multiplier."""
    assert multiply_schedules()(7, 100) == 1.0


def test_zero_total_steps():
    """Edge case: total_steps=0 should not crash."""
    assert polynomial(0, 0) == 1.0  # Step 0 = full LR.
    assert cosine(0, 0) == 1.0
    assert trapezoidal(0, 0) == 1.0


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
