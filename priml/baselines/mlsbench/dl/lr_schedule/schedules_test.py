"""Tests for the ``dl-lr-schedule`` schedules, against their closed forms."""

from __future__ import annotations

import math

import pytest

from priml.baselines.mlsbench.dl.lr_schedule.schedules import (
    constant,
    cosine,
    one_cycle,
    warmup_cosine,
)


def test_constant_ignores_the_epoch() -> None:
    assert {constant(e, base_lr=0.3, total_epochs=7) for e in range(7)} == {0.3}


def test_cosine_starts_at_the_base_rate_and_halves_at_the_midpoint() -> None:
    assert cosine(0, base_lr=0.4, total_epochs=8) == 0.4
    assert cosine(4, base_lr=0.4, total_epochs=8) == pytest.approx(0.2)
    assert cosine(8, base_lr=0.4, total_epochs=8) == pytest.approx(0.0)


def test_warmup_ramps_linearly_then_anneals_from_the_peak() -> None:
    rates = [
        warmup_cosine(e, base_lr=1.0, total_epochs=9, warmup_epochs=4) for e in range(9)
    ]
    assert rates[:4] == [0.25, 0.5, 0.75, 1.0]
    assert rates[4] == 1.0
    assert rates[-1] == pytest.approx(0.5 * (1 + math.cos(math.pi * 4 / 5)))


def test_one_cycle_rises_to_the_peak_then_falls_to_the_floor() -> None:
    def rate(epoch: int) -> float:
        return one_cycle(
            epoch,
            base_lr=1.0,
            total_epochs=11,
            pct_start=0.3,
            div_factor=4.0,
            final_div=8.0,
        )

    assert rate(0) == pytest.approx(0.25)
    assert rate(3) == pytest.approx(1.0)
    assert rate(10) == pytest.approx(0.125)
    assert rate(2) < rate(3) > rate(4)


def test_one_cycle_of_one_epoch_stays_at_the_start() -> None:
    assert one_cycle(
        0,
        base_lr=1.0,
        total_epochs=1,
        pct_start=0.3,
        div_factor=4.0,
        final_div=8.0,
    ) == pytest.approx(0.25)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
