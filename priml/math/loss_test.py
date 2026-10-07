"""Tests for pure loss functions."""

from __future__ import annotations

import math

from torch import nn

import pytest
import torch

from priml.math.loss import (
    cross_entropy_logz,
    cross_entropy_with_batched_smoothing,
    log_stablemax,
    stablemax_cross_entropy,
)


def test_cross_entropy_logz_is_cross_entropy_and_its_normalizer() -> None:
    logits, target = torch.randn(2, 3, 5), torch.randint(0, 5, (2, 3))
    nll, logz = cross_entropy_logz(logits, target)
    expected = nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten())
    torch.testing.assert_close(nll.mean(), expected)
    torch.testing.assert_close(logz, logits.logsumexp(-1))
    assert nll.shape == logz.shape == (2, 3)


def test_cross_entropy_logz_restricts_the_softmax_to_finite_logits() -> None:
    logits, target = torch.randn(4, 5), torch.tensor([0, 2, 2, 4])
    allowed = torch.tensor([True, False, True, False, True])
    nll, logz = cross_entropy_logz(logits.masked_fill(~allowed, -math.inf), target)
    kept = logits[:, allowed]
    torch.testing.assert_close(logz, kept.logsumexp(-1))
    torch.testing.assert_close(nll, logz - logits.gather(-1, target[:, None])[:, 0])


def test_cross_entropy_logz_gradient_is_softmax_minus_onehot() -> None:
    logits = torch.randn(3, 5, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    nll, _ = cross_entropy_logz(logits, target)
    (gradient,) = torch.autograd.grad(nll.sum(), logits)
    onehot = nn.functional.one_hot(target, 5).float()
    torch.testing.assert_close(gradient, logits.softmax(-1).detach() - onehot)


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


def test_scalar_smoothing_delegates_to_f_cross_entropy() -> None:
    torch.manual_seed(3)
    logits = torch.randn(6, 4)
    target = torch.randint(0, 4, (6,))
    actual = cross_entropy_with_batched_smoothing(logits, target, label_smoothing=0.2)
    expected = nn.functional.cross_entropy(logits, target, label_smoothing=0.2)
    torch.testing.assert_close(actual, expected)


def test_batched_smoothing_none_and_sum_reductions_agree_with_mean() -> None:
    torch.manual_seed(4)
    logits = torch.randn(5, 3)
    target = torch.tensor([0, 1, 2, -100, 1])
    smoothing = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    per_element = cross_entropy_with_batched_smoothing(
        logits,
        target,
        label_smoothing=smoothing,
        reduction="none",
    )
    total = cross_entropy_with_batched_smoothing(
        logits,
        target,
        label_smoothing=smoothing,
        reduction="sum",
    )
    mean = cross_entropy_with_batched_smoothing(
        logits,
        target,
        label_smoothing=smoothing,
    )
    assert per_element.shape == (5,)
    assert per_element[3] == 0.0
    torch.testing.assert_close(total, per_element.sum())
    torch.testing.assert_close(mean, total / 4)


def test_batched_smoothing_weighted_rank_three_uses_class_axis_and_target_weights() -> (
    None
):
    logits = torch.tensor(
        [
            [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0], [1.0, 0.0, 3.0, 2.0]],
            [[2.0, 0.0, 1.0, 3.0], [0.0, 3.0, 2.0, 1.0], [1.0, 2.0, 0.0, 3.0]],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor([[0, 1, 2, 3], [2, 0, 1, 3]])
    smoothing = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.3, 0.2, 0.1, 0.4]])
    weight = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    actual = cross_entropy_with_batched_smoothing(
        logits,
        target,
        weight=weight,
        ignore_index=3,
        label_smoothing=smoothing,
    )
    log_probs = logits.log_softmax(dim=1)
    safe_target = target.clamp(0, 2)
    gathered = log_probs.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    target_weight = weight[safe_target]
    # cross_entropy_with_batched_smoothing broadcasts weights across batch/spatial axes.
    smooth = -(log_probs * weight.view(1, 3, 1)).sum(dim=1) / 3
    mask = (target != 3).to(logits.dtype)
    losses = ((1 - smoothing) * -target_weight * gathered + smoothing * smooth) * mask
    expected = losses.sum() / (target_weight * mask).sum()
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
    x = torch.full((2, 3), 2e38, dtype=torch.float32)
    logp = log_stablemax(x)
    assert torch.isfinite(logp).all(), logp
    # Three equal logits: each gets a third of the mass.
    torch.testing.assert_close(logp, torch.full_like(logp, -math.log(3.0)))


def test_stablemax_cross_entropy_normalizes_over_classes() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0], [3.0, 0.0, 1.0]])
    labels = torch.tensor([2, 0])
    actual = stablemax_cross_entropy(logits, labels)
    expected = (
        -log_stablemax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    )
    torch.testing.assert_close(actual, expected)


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


def test_scalar_smoothing_forwards_weight_ignore_and_reduction() -> None:
    logits = torch.tensor(
        [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [1.0, 3.0, 2.0]],
    )
    target = torch.tensor([2, 9, 0])
    weight = torch.tensor([0.5, 1.5, 2.0])

    actual = cross_entropy_with_batched_smoothing(
        logits,
        target,
        weight=weight,
        ignore_index=9,
        reduction="sum",
        label_smoothing=0.25,
    )
    expected = nn.functional.cross_entropy(
        logits,
        target,
        weight=weight,
        ignore_index=9,
        reduction="sum",
        label_smoothing=0.25,
    )

    torch.testing.assert_close(actual, expected)


def test_scalar_smoothing_defaults_to_zero() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    target = torch.tensor([2, 0])

    actual = cross_entropy_with_batched_smoothing(logits, target)
    expected = nn.functional.cross_entropy(logits, target, label_smoothing=0.0)

    torch.testing.assert_close(actual, expected)


def test_tensor_smoothing_unweighted_uses_class_axis_and_negative_logprob() -> None:
    logits = torch.tensor(
        [
            [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0], [1.0, 0.0, 3.0, 2.0]],
            [[2.0, 0.0, 1.0, 3.0], [0.0, 3.0, 2.0, 1.0], [1.0, 2.0, 0.0, 3.0]],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor([[0, 1, -100, 2], [2, 0, 1, -100]])
    smoothing = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.3, 0.2, 0.1, 0.4]],
        dtype=torch.float64,
    )
    log_probs = logits.log_softmax(dim=1)
    safe_target = target.clamp(0, 2)
    gathered = log_probs.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    mask = (target != -100).to(logits.dtype)
    expected_elements = (
        (1 - smoothing) * -gathered + smoothing * -log_probs.mean(dim=1)
    ) * mask
    expected = expected_elements.sum() / mask.sum()

    actual = cross_entropy_with_batched_smoothing(
        logits,
        target,
        label_smoothing=smoothing,
    )

    torch.testing.assert_close(actual, expected)


def test_tensor_smoothing_mean_keeps_a_single_valid_element() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [1.0, 3.0, 2.0]])
    target = torch.tensor([1, -100, -100])
    smoothing = torch.tensor([0.25, 0.5, 0.75])
    per_element = cross_entropy_with_batched_smoothing(
        logits,
        target,
        label_smoothing=smoothing,
        reduction="none",
    )
    mean = cross_entropy_with_batched_smoothing(
        logits,
        target,
        label_smoothing=smoothing,
    )

    assert per_element[0] > 0
    assert per_element[1:].eq(0).all()
    torch.testing.assert_close(mean, per_element[0])


def test_stablemax_cross_entropy_handles_rank_three_logits() -> None:
    logits = torch.tensor(
        [
            [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0], [1.0, 0.0, 3.0, 2.0]],
            [[2.0, 0.0, 1.0, 3.0], [0.0, 3.0, 2.0, 1.0], [1.0, 2.0, 0.0, 3.0]],
        ],
    )
    labels = torch.tensor([[0, 1, 3], [2, 0, 1]], dtype=torch.int32)
    valid_mask = torch.tensor([[True, False, True], [True, True, False]])
    logprobs = log_stablemax(logits, dim=-1)
    expected = -torch.where(
        valid_mask,
        logprobs.gather(-1, labels.to(torch.long).unsqueeze(-1)).squeeze(-1),
        0.0,
    )

    actual = stablemax_cross_entropy(logits, labels, valid_mask=valid_mask)

    assert actual.shape == (2, 3)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("ignored", [False, True])
def test_tensor_smoothing_matches_scalar_with_subunit_weight(ignored: bool) -> None:
    logits = torch.zeros(2, 3, dtype=torch.float64)
    target = torch.full((2,), -100) if ignored else torch.tensor([0, 2])
    weight = torch.full((3,), 0.1, dtype=logits.dtype)
    actual = cross_entropy_with_batched_smoothing(
        logits,
        target,
        weight=weight,
        label_smoothing=torch.full((2,), 0.2),
    )
    expected = nn.functional.cross_entropy(
        logits,
        target,
        weight=weight,
        label_smoothing=0.2,
    )
    torch.testing.assert_close(actual, expected, equal_nan=True)


@pytest.mark.parametrize("target", [torch.tensor([0, 3]), torch.tensor([-1, 0])])
def test_tensor_smoothing_rejects_invalid_targets(target: torch.Tensor) -> None:
    with pytest.raises((IndexError, RuntimeError), match=r"(bounds|range)"):
        cross_entropy_with_batched_smoothing(
            torch.zeros(2, 3),
            target,
            label_smoothing=torch.zeros(2),
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
