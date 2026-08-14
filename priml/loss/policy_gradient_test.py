"""Tests for the clipped policy-gradient objective."""

from __future__ import annotations

from typing import TypedDict, Unpack

import math

from torch import Tensor

import pytest
import torch

from priml.loss.policy_gradient import (
    ClippedPolicyLoss,
    categorical_entropy,
    clipped_policy_loss,
)


class _TermsOverrides(TypedDict, total=False):
    log_probs: Tensor
    behavior_log_probs: Tensor
    advantages: Tensor
    values: Tensor


def _terms(**overrides: Unpack[_TermsOverrides]) -> ClippedPolicyLoss:
    return clipped_policy_loss(
        log_probs=overrides.get(
            "log_probs",
            torch.tensor([-0.70, -0.60, -0.80, -0.50]),
        ),
        behavior_log_probs=overrides.get(
            "behavior_log_probs",
            torch.tensor([-0.70, -0.60, -0.80, -0.50]),
        ),
        advantages=overrides.get(
            "advantages",
            torch.tensor([1.0, -0.5, 0.25, 0.5]),
        ),
        values=overrides.get(
            "values",
            torch.tensor([0.1, 0.2, -0.1, 0.0]),
        ),
        behavior_values=torch.tensor([0.1, 0.2, -0.1, 0.0]),
        targets=torch.tensor([0.5, 0.1, 0.4, -0.2]),
        entropy=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        clip_epsilon=0.2,
    )


def test_unchanged_policy_gives_unit_ratio_and_zero_divergence() -> None:
    terms = _terms()
    assert float(terms.approx_kl) == pytest.approx(0.0)
    assert float(terms.clip_fraction) == 0.0
    # With ratio 1 the policy term is the negated mean standardized advantage,
    # which is zero by construction.
    assert float(terms.policy) == pytest.approx(0.0, abs=1e-6)


def test_value_term_is_half_mean_squared_error_inside_the_trust_region() -> None:
    terms = _terms()
    values = torch.tensor([0.1, 0.2, -0.1, 0.0])
    targets = torch.tensor([0.5, 0.1, 0.4, -0.2])
    assert float(terms.value) == pytest.approx(
        float(0.5 * ((values - targets) ** 2).mean()),
    )


def test_clipping_bounds_the_value_term_against_a_far_prediction() -> None:
    # A value 10 away from its behavior estimate is clipped to 0.2 away, so the
    # squared error is bounded by the unclipped one.
    clipped = _terms(values=torch.tensor([10.0, 0.2, -0.1, 0.0]))
    assert float(clipped.value) < float(0.5 * (10.0 - 0.5) ** 2)


def test_ratio_leaving_the_trust_region_is_reported() -> None:
    terms = _terms(log_probs=torch.tensor([-0.70, -0.60, -0.80, 0.50]))
    assert float(terms.clip_fraction) == pytest.approx(0.25)
    assert float(terms.approx_kl) > 0.0


def test_divergence_estimate_is_non_negative_in_both_directions() -> None:
    behavior = torch.tensor([-0.7, -0.6, -0.8, -0.5])
    for shift in (-0.3, 0.3):
        terms = _terms(log_probs=behavior + shift, behavior_log_probs=behavior)
        assert float(terms.approx_kl) > 0.0


def test_constant_advantages_stay_finite() -> None:
    # Zero spread would divide by zero without the epsilon floor.
    terms = _terms(advantages=torch.ones(4))
    assert math.isfinite(float(terms.policy))


def test_gradients_reach_both_heads() -> None:
    log_probs = torch.tensor([-0.7, -0.6, -0.8, -0.5], requires_grad=True)
    values = torch.tensor([0.1, 0.2, -0.1, 0.0], requires_grad=True)
    terms = _terms(log_probs=log_probs, values=values)
    (terms.policy + terms.value).backward()
    assert log_probs.grad is not None
    assert values.grad is not None
    assert float(log_probs.grad.abs().sum()) > 0.0
    assert float(values.grad.abs().sum()) > 0.0


def test_entropy_matches_closed_form_for_a_uniform_distribution() -> None:
    log_probs = torch.full((2, 5), -math.log(5.0))
    assert categorical_entropy(log_probs).tolist() == pytest.approx(
        [math.log(5.0)] * 2,
    )


def test_entropy_of_a_deterministic_distribution_is_zero() -> None:
    logits = torch.tensor([[0.0, -math.inf, -math.inf]])
    entropy = categorical_entropy(torch.log_softmax(logits, dim=-1))
    assert float(entropy) == pytest.approx(0.0)


def test_masked_actions_do_not_poison_entropy() -> None:
    # exp(-inf) * -inf is NaN, so a masked action must be dropped, not summed.
    logits = torch.tensor([[0.0, 0.0, -math.inf]])
    entropy = categorical_entropy(torch.log_softmax(logits, dim=-1))
    assert float(entropy) == pytest.approx(math.log(2.0))


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
