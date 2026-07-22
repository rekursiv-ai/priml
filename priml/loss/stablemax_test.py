"""Tests for stablemax cross-entropy."""

from __future__ import annotations

import torch

from priml.loss.stablemax import (
    log_stablemax,
    stablemax_cross_entropy,
)


def test_log_stablemax_normalizes() -> None:
    x = torch.randn(3, 5, dtype=torch.float64)
    logp = log_stablemax(x, dim=-1)
    # exp(logp) should sum to 1 per row.
    sums = logp.exp().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-12)


def test_stablemax_cross_entropy_zero_loss_on_perfect_logits() -> None:
    # Logits favoring the correct class infinitely-strongly -> tiny loss.
    logits = torch.full((2, 4), -10.0)
    labels = torch.tensor([1, 2])
    logits[0, 1] = 10.0
    logits[1, 2] = 10.0
    loss = stablemax_cross_entropy(logits, labels)
    assert loss.shape == labels.shape
    assert loss.dtype == torch.float64
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


def test_stablemax_cross_entropy_returns_fp64_from_bf16_logits() -> None:
    # Under autocast logits arrive in bf16; the function must upcast.
    logits = torch.randn(2, 3, dtype=torch.bfloat16)
    labels = torch.tensor([0, 1])
    loss = stablemax_cross_entropy(logits, labels)
    assert loss.dtype == torch.float64


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
