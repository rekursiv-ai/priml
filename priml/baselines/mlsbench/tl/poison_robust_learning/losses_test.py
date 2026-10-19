"""Tests for the published robust losses of ``security-poison-robust-learning``.

Each is pinned to its closed form on the scaffold's fixed scores, value and
gradient, and at its degenerate setting to the cross-entropy it extends.
"""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional

import pytest
import torch

from priml.baselines.mlsbench.dl.vision.train_step import cross_entropy
from priml.baselines.mlsbench.dl.vision.train_step_test import CONTEXT, logits
from priml.baselines.mlsbench.tl.poison_robust_learning.losses import (
    bootstrap,
    generalized_ce,
    symmetric_ce,
)


def probabilities() -> tuple[Tensor, Tensor]:
    """Return the fixed scores' softmax and their labels."""
    scores, label = logits()
    return functional.softmax(scores, dim=-1), label


def test_generalized_ce_is_the_mean_of_one_minus_p_to_the_q_over_q() -> None:
    scores, label = logits()
    p = probabilities()[0][torch.arange(3), label]
    loss = generalized_ce(
        scores,
        label=label,
        context=CONTEXT,
        q=0.7,
        min_probability=1e-8,
    )
    assert loss["loss"] == pytest.approx(float(((1 - p**0.7) / 0.7).mean()), rel=1e-6)


def test_generalized_ce_weights_the_nll_gradient_by_p_to_the_q() -> None:
    scores, label = logits()
    leaf = scores.clone().requires_grad_()
    generalized_ce(leaf, label=label, context=CONTEXT, q=0.7, min_probability=1e-8)[
        "loss"
    ].backward()
    probs = probabilities()[0]
    p = probs[torch.arange(3), label]
    nll_grad = probs - functional.one_hot(label, 4)
    expected = p.unsqueeze(1) ** 0.7 * nll_grad / 3
    assert leaf.grad is not None
    assert torch.allclose(leaf.grad, expected, atol=1e-7)


def test_generalized_ce_floors_the_true_class_probability() -> None:
    scores = torch.tensor([[60.0, -60.0, 0.0], [0.0, 1.0, 2.0]])
    label = torch.tensor([1, 2])
    loss = generalized_ce(
        scores,
        label=label,
        context=CONTEXT,
        q=0.5,
        min_probability=0.25,
    )
    p = torch.tensor([0.25, float(functional.softmax(scores[1], dim=0)[2])])
    assert loss["loss"] == pytest.approx(float(((1 - p**0.5) / 0.5).mean()), rel=1e-6)


def test_symmetric_ce_adds_beta_times_the_reverse_cross_entropy() -> None:
    scores, label = logits()
    probs = probabilities()[0]
    ce = functional.cross_entropy(scores, label)
    log_target = torch.log(functional.one_hot(label, 4).float().clamp_min(1e-4))
    rce = -(probs * log_target).sum(dim=1).mean()
    loss = symmetric_ce(
        scores,
        label=label,
        context=CONTEXT,
        alpha=1.0,
        beta=0.5,
        min_probability=1e-8,
        min_target=1e-4,
    )
    assert loss["loss"] == pytest.approx(float(ce + 0.5 * rce), rel=1e-6)


def test_symmetric_ce_with_beta_zero_is_alpha_times_cross_entropy() -> None:
    scores, label = logits()
    plain = cross_entropy(scores, label=label, context=CONTEXT)["loss"]
    loss = symmetric_ce(
        scores,
        label=label,
        context=CONTEXT,
        alpha=2.0,
        beta=0.0,
        min_probability=1e-8,
        min_target=1e-4,
    )
    assert loss["loss"] == 2.0 * plain


def test_bootstrap_targets_mix_the_label_with_the_detached_prediction() -> None:
    scores, label = logits()
    probs = probabilities()[0]
    target = 0.8 * functional.one_hot(label, 4) + 0.2 * probs
    expected = -(target * functional.log_softmax(scores, dim=1)).sum(dim=1).mean()
    loss = bootstrap(scores, label=label, context=CONTEXT, beta=0.8)["loss"]
    assert loss == pytest.approx(float(expected), rel=1e-6)


def test_bootstrap_backpropagates_through_the_log_probabilities_only() -> None:
    scores, label = logits()
    leaf = scores.clone().requires_grad_()
    bootstrap(leaf, label=label, context=CONTEXT, beta=0.8)["loss"].backward()
    probs = probabilities()[0]
    target = 0.8 * functional.one_hot(label, 4) + 0.2 * probs
    assert leaf.grad is not None
    assert torch.allclose(leaf.grad, (probs - target) / 3, atol=1e-7)


def test_bootstrap_with_beta_one_is_cross_entropy() -> None:
    scores, label = logits()
    plain = cross_entropy(scores, label=label, context=CONTEXT)["loss"]
    loss = bootstrap(scores, label=label, context=CONTEXT, beta=1.0)["loss"]
    assert loss == pytest.approx(float(plain), rel=1e-6)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
