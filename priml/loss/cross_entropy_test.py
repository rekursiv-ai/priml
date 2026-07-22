"""Tests for cross_entropy_with_batched_smoothing."""

from __future__ import annotations

from torch import nn

import torch

from priml.loss.cross_entropy import cross_entropy_with_batched_smoothing


def test_batched_smoothing_4d_matches_f_cross_entropy() -> None:
    """LOSSOPT-001: rank>2 logits must use class dim=1, not dim=-1."""
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 3, 4)
    target = torch.randint(0, 5, (2, 3, 4))
    smoothing = torch.full_like(target, 0.0, dtype=torch.float)

    actual = cross_entropy_with_batched_smoothing(
        logits,
        target,
        label_smoothing=smoothing,
    )
    expected = nn.functional.cross_entropy(logits, target, label_smoothing=0.0)
    torch.testing.assert_close(actual, expected)


def test_batched_smoothing_weighted_matches_f_cross_entropy() -> None:
    """LOSSOPT-002: weighted smoothing must match F.cross_entropy."""
    torch.manual_seed(1)
    logits = torch.randn(8, 4)
    target = torch.randint(0, 4, (8,))
    weight = torch.tensor([0.5, 1.0, 2.0, 1.5])
    smoothing = torch.full_like(target, 0.1, dtype=torch.float)

    actual = cross_entropy_with_batched_smoothing(
        logits,
        target,
        weight=weight,
        label_smoothing=smoothing,
    )
    expected = nn.functional.cross_entropy(
        logits,
        target,
        weight=weight,
        label_smoothing=0.1,
    )
    torch.testing.assert_close(actual, expected)


def test_batched_smoothing_out_of_range_ignore_index() -> None:
    """LOSSOPT-003: positive ignore_index >= C must not crash on weight index."""
    torch.manual_seed(2)
    logits = torch.randn(4, 3)
    target = torch.tensor([0, 255, 2, 255])
    weight = torch.tensor([1.0, 1.0, 1.0])
    smoothing = torch.full_like(target, 0.1, dtype=torch.float)

    actual = cross_entropy_with_batched_smoothing(
        logits,
        target,
        weight=weight,
        ignore_index=255,
        label_smoothing=smoothing,
    )
    expected = nn.functional.cross_entropy(
        logits,
        target,
        weight=weight,
        ignore_index=255,
        label_smoothing=0.1,
    )
    torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
