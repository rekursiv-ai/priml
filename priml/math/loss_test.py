"""Tests for pure loss functions."""

from __future__ import annotations

import math

from torch import nn

import torch

from priml.math.loss import (
    cross_entropy_with_batched_smoothing,
    log_stablemax,
    stablemax_cross_entropy,
)


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


def test_log_stablemax_normalizes() -> None:
    x = torch.randn(3, 5, dtype=torch.float64)
    logp = log_stablemax(x, dim=-1)
    # exp(logp) should sum to 1 per row.
    sums = logp.exp().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-12)


def test_log_stablemax_matches_the_literal_definition() -> None:
    """Pins the docstring's derivation: log_softmax . log_modulus == log(s/sum s)."""
    x = torch.randn(4, 7, dtype=torch.float64) * 3.0
    surrogate = torch.where(x < 0, 1.0 / (1.0 - x), x + 1.0)
    expected = (surrogate / surrogate.sum(dim=-1, keepdim=True)).log()
    torch.testing.assert_close(log_stablemax(x), expected)


def test_log_stablemax_gradient_is_finite_in_fp32() -> None:
    """A branch reaching log1p(-1) = -inf poisons the gradient with NaN."""
    x = torch.tensor([[1.0, 0.5, -0.5]], dtype=torch.float32, requires_grad=True)
    log_stablemax(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), x.grad


def test_log_stablemax_survives_logits_that_overflow_the_sum() -> None:
    """``s(x)`` grows linearly, so summing it overflows fp32 before ``x`` does.

    Normalizing in log space keeps the answer exact where forming ``s(x)`` and
    dividing yields ``-inf`` everywhere.
    """
    x = torch.full((1, 3), 2e38, dtype=torch.float32)
    logp = log_stablemax(x)
    assert torch.isfinite(logp).all(), logp
    # Three equal logits: each gets a third of the mass.
    torch.testing.assert_close(logp, torch.full_like(logp, -math.log(3.0)))


def test_stablemax_cross_entropy_zero_loss_on_perfect_logits() -> None:
    # Logits favoring the correct class infinitely-strongly -> tiny loss.
    logits = torch.full((2, 4), -10.0)
    labels = torch.tensor([1, 2])
    logits[0, 1] = 10.0
    logits[1, 2] = 10.0
    loss = stablemax_cross_entropy(logits, labels)
    assert loss.shape == labels.shape
    assert (loss < 0.1).all()


def test_stablemax_cross_entropy_ignore_index_masks_loss() -> None:
    logits = torch.randn(3, 5)
    labels = torch.tensor([0, -100, 2])
    loss = stablemax_cross_entropy(logits, labels)
    # Position 1 must contribute exactly zero to the loss.
    assert loss[1].item() == 0.0
    # Others must be positive.
    assert loss[0].item() > 0.0
    assert loss[2].item() > 0.0


def test_stablemax_cross_entropy_preserves_the_logits_dtype() -> None:
    """Log-space normalization is accurate at the input's own precision.

    Upcasting to fp64 would refine a value already wrong in the third decimal
    from bf16 rounding, so the dtype stays the caller's to choose.
    """
    labels = torch.tensor([0, 1])
    for dtype in (torch.bfloat16, torch.float32, torch.float64):
        logits = torch.randn(2, 3, dtype=dtype)
        assert stablemax_cross_entropy(logits, labels).dtype == dtype


def test_stablemax_cross_entropy_explicit_valid_mask_overrides_ignore() -> None:
    logits = torch.randn(3, 4)
    labels = torch.tensor([0, 1, 2])
    mask = torch.tensor([True, False, True])
    loss = stablemax_cross_entropy(logits, labels, valid_mask=mask)
    assert loss[1].item() == 0.0
    assert loss[0].item() > 0.0
    assert loss[2].item() > 0.0


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
