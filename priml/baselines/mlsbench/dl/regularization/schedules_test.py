"""Tests for the VGG warmup-then-cosine schedule.

Its bits against the script's ``LambdaLR`` are proven by the capture, which
compares all 200 rates at the ``vgg16bn-cifar100`` setting.
"""

from __future__ import annotations

import math

from priml.baselines.mlsbench.dl.regularization.schedules import (
    warmup_cosine_factor,
)


def rate(epoch: int) -> float:
    """Return the schedule at a 0.1 base rate, 7 epochs, 3 of warmup."""
    return warmup_cosine_factor(epoch, base_lr=0.1, total_epochs=7, warmup_epochs=3)


def test_warmup_reaches_the_base_rate_in_its_last_epoch() -> None:
    assert [rate(e) for e in range(3)] == [0.1 * (1 / 3), 0.1 * (2 / 3), 0.1 * 1.0]


def test_the_cosine_starts_at_the_base_rate_and_follows_half_a_period() -> None:
    assert rate(3) == 0.1
    assert rate(5) == 0.1 * (0.5 * (1.0 + math.cos(math.pi * 0.5)))


def test_a_run_no_longer_than_its_warmup_divides_by_one() -> None:
    assert warmup_cosine_factor(3, base_lr=0.2, total_epochs=3, warmup_epochs=3) == 0.2


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
