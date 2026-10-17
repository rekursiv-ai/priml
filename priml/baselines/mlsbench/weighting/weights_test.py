"""Tests for the ``cv-sample-weighting`` class weights, against their closed forms."""

from __future__ import annotations

import torch

from priml.baselines.mlsbench.weighting.weights import (
    effective_number,
    inverse_frequency,
    log_frequency_gap,
    uniform,
)


def test_uniform_weighs_every_class_one() -> None:
    assert torch.equal(uniform(torch.tensor([4.0, 1.0]), num_classes=3), torch.ones(3))


def test_inverse_frequency_is_the_damped_inverse_rescaled_to_the_class_count() -> None:
    weights = inverse_frequency(torch.tensor([4.0, 1.0]), num_classes=2)
    # sqrt(5 / 8) : sqrt(5 / 2) is 1 : 2, rescaled to sum to 2.
    torch.testing.assert_close(weights, torch.tensor([2 / 3, 4 / 3]))


def test_effective_number_approaches_inverse_frequency_as_beta_nears_one() -> None:
    counts = torch.tensor([4.0, 1.0])
    weights = effective_number(counts, num_classes=2, beta=0.5)
    # (1 - b) / (1 - b^n): 0.5 / 0.9375 and 1, square-rooted, rescaled to 2.
    raw = torch.tensor([(0.5 / 0.9375) ** 0.5, 1.0])
    torch.testing.assert_close(weights, raw / raw.sum() * 2)


def test_log_frequency_gap_ramps_from_one_to_one_plus_spread() -> None:
    weights = log_frequency_gap(
        torch.tensor([4.0, 2.0, 1.0]),
        num_classes=3,
        spread=4.0,
    )
    raw = torch.tensor([1.0, 3.0, 5.0])
    torch.testing.assert_close(weights, raw / raw.sum() * 3)


def test_log_frequency_gap_of_equal_counts_is_uniform() -> None:
    weights = log_frequency_gap(torch.full((3,), 2.0), num_classes=3, spread=4.0)
    assert torch.equal(weights, torch.ones(3))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
