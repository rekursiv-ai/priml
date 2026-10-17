"""Tests for the ACT sequence and halt objective."""

from __future__ import annotations

import pytest
import torch

from priml.loss.act import ACTLoss
from priml.loss.token import StablemaxTokens
from priml.math.loss import log_stablemax_via_probabilities


def test_act_loss_matches_source_order_values_and_gradients() -> None:
    """Divide each sample's tokens before summing and preserve empty masks."""
    logits = torch.tensor(
        [
            [
                [-4.0, 0.25, 3.0, -0.25],
                [2.0, -1.0, 0.5, -3.0],
                [0.1, 0.2, -0.7, 1.5],
                [-2.0, 0.75, 1.25, -0.5],
            ],
            [
                [1.5, -3.0, 0.2, 0.5],
                [-0.4, 1.1, 2.5, -2.0],
                [0.0, -0.5, 0.75, 3.0],
                [0.25, 1.75, -1.5, -0.25],
            ],
            [
                [-1.0, 2.5, 0.25, -3.0],
                [1.25, -2.5, 3.0, -0.75],
                [-0.2, 0.3, 1.4, -1.1],
                [2.25, -1.75, 0.6, -0.4],
            ],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = torch.tensor(
        [[2, 0, -100, -100], [-100, -100, -100, -100], [-100, 2, -100, -100]],
    )
    halt_logits = torch.tensor(
        [-0.3, 0.4, 0.2],
        dtype=torch.float64,
        requires_grad=True,
    )
    stable = torch.where(
        logits < 0,
        1 / (1 - logits + 1e-30),
        logits + 1,
    )
    log_probabilities = (stable / stable.sum(dim=-1, keepdim=True)).log()
    counted = labels != -100
    safe_labels = labels.masked_fill(~counted, 0)
    selected = log_probabilities.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_loss = -torch.where(counted, selected, 0.0)
    counts = counted.sum(dim=-1)
    normalized = token_loss / counts.clamp_min(1).unsqueeze(-1)
    sequence_loss = normalized.sum(dim=-1)
    lm_loss = normalized.sum()
    correct = ((logits.argmax(dim=-1) == labels) | ~counted).all(dim=-1)
    halt_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        halt_logits,
        correct.to(halt_logits.dtype),
        reduction="sum",
    )
    expected = lm_loss + 0.5 * halt_loss

    objective = ACTLoss.Config(
        token_loss=StablemaxTokens.Config(
            log_normalizer=log_stablemax_via_probabilities,
        ),
    ).make()
    actual = objective(logits=logits, labels=labels, halt_logits=halt_logits)

    assert torch.equal(actual.sequence_loss, sequence_loss)
    assert torch.equal(actual.lm_loss, lm_loss)
    assert torch.equal(actual.halt_loss, halt_loss)
    assert torch.equal(actual.correct, correct)
    assert torch.equal(actual.loss_counts, counts)
    assert torch.equal(actual.loss, expected)
    actual_gradients = torch.autograd.grad(actual.loss, (logits, halt_logits))
    expected_gradients = torch.autograd.grad(expected, (logits, halt_logits))
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        assert torch.equal(actual_gradient, expected_gradient)


def test_act_loss_excludes_inactive_rows_from_losses_and_gradients() -> None:
    """Inactive streaming slots contribute no token, halt, or score values."""
    logits = torch.tensor(
        [
            [[1.0, -0.5, 0.25], [-0.25, 1.5, 0.0]],
            [[-1.0, 2.0, 0.5], [1.0, -0.5, 0.25]],
            [[0.5, -0.25, 1.0], [1.25, 0.0, -0.5]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = torch.tensor([[0, 1], [1, 0], [2, 0]])
    halt_logits = torch.tensor(
        [0.1, -0.2, 0.3],
        dtype=torch.float64,
        requires_grad=True,
    )
    active = torch.tensor([True, False, True])
    objective = ACTLoss.Config().make()

    actual = objective(
        logits=logits,
        labels=labels,
        halt_logits=halt_logits,
        active=active,
    )
    expected = objective(
        logits=logits[active],
        labels=labels[active],
        halt_logits=halt_logits[active],
    )

    assert torch.equal(actual.loss, expected.loss)
    assert torch.equal(actual.lm_loss, expected.lm_loss)
    assert torch.equal(actual.halt_loss, expected.halt_loss)
    assert actual.loss_counts.tolist() == [2, 0, 2]
    assert actual.correct.tolist() == [True, False, True]
    logits_gradient, halt_gradient = torch.autograd.grad(
        actual.loss,
        (logits, halt_logits),
    )
    assert torch.equal(logits_gradient[1], torch.zeros_like(logits_gradient[1]))
    assert halt_gradient[1] == 0


@pytest.mark.parametrize("halt_weight", [-0.5, float("nan")])
def test_act_loss_rejects_invalid_halt_weight(halt_weight: float) -> None:
    """Reject negative and non-finite halt coefficients at construction."""
    with pytest.raises(ValueError, match="halt_weight"):
        ACTLoss.Config(halt_weight=halt_weight).make()


def test_act_loss_rejects_mismatched_sequence_shapes() -> None:
    """Require token targets to match the logits' batch and sequence axes."""
    objective = ACTLoss.Config().make()
    with pytest.raises(
        ValueError,
        match=r"\AACTLoss expects logits \[B,S,V\] and labels \[B,S\] with matching leading dimensions\.\Z",
    ):
        objective(
            logits=torch.zeros(2, 3, 4),
            labels=torch.zeros(2, 4, dtype=torch.long),
            halt_logits=torch.zeros(2),
        )


def test_act_loss_rejects_mismatched_halt_shape() -> None:
    """Require one halt logit for each batch row."""
    objective = ACTLoss.Config().make()
    with pytest.raises(
        ValueError,
        match=r"\AACTLoss expects halt_logits with shape \[B\]\.\Z",
    ):
        objective(
            logits=torch.zeros(2, 3, 4),
            labels=torch.zeros(2, 3, dtype=torch.long),
            halt_logits=torch.zeros(3),
        )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
