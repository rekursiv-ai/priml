"""Tests for the published losses of ``cv-classification-loss``.

Each is pinned to its closed form on the scaffold's fixed scores, and its
setting at zero to the pristine cross-entropy it extends.
"""

from __future__ import annotations

from torch.nn import functional

import pytest
import torch

from priml.baselines.mlsbench.dl.classification_loss.losses import (
    focal,
    label_smoothing,
    poly,
)
from priml.baselines.mlsbench.dl.vision.train_step import cross_entropy
from priml.baselines.mlsbench.dl.vision.train_step_test import CONTEXT, logits


def test_label_smoothing_mixes_the_nll_with_the_uniform_nll() -> None:
    scores, label = logits()
    log_probs = functional.log_softmax(scores, dim=-1)
    nll = -log_probs[torch.arange(3), label].mean()
    uniform = -log_probs.mean(dim=-1).mean()
    loss = label_smoothing(scores, label=label, context=CONTEXT, smoothing=0.1)["loss"]
    assert loss == pytest.approx(float(0.9 * nll + 0.1 * uniform), rel=1e-6)


def test_label_smoothing_pulls_the_probabilities_toward_uniform() -> None:
    scores, label = logits()
    leaf = scores.clone().requires_grad_()
    label_smoothing(leaf, label=label, context=CONTEXT, smoothing=0.1)[
        "loss"
    ].backward()
    target = 0.9 * functional.one_hot(label, 4) + 0.1 / 4
    expected = (functional.softmax(scores, dim=-1) - target) / 3
    assert leaf.grad is not None
    assert torch.allclose(leaf.grad, expected, atol=1e-7)


def test_focal_loss_weights_each_example_by_one_minus_its_probability() -> None:
    scores, label = logits()
    probability = functional.softmax(scores, dim=-1)[torch.arange(3), label]
    nll = -probability.log()
    loss = focal(scores, label=label, context=CONTEXT, gamma=2.0)["loss"]
    assert loss == pytest.approx(float(((1 - probability) ** 2 * nll).mean()), rel=1e-6)


def test_focal_loss_with_gamma_zero_is_the_mean_of_the_example_losses() -> None:
    scores, label = logits()
    expected = functional.cross_entropy(scores, label, reduction="none").mean()
    assert focal(scores, label=label, context=CONTEXT, gamma=0.0)["loss"] == expected


def test_poly_loss_adds_epsilon_times_one_minus_the_probability() -> None:
    scores, label = logits()
    probability = functional.softmax(scores, dim=-1)[torch.arange(3), label]
    nll = -probability.log().mean()
    loss = poly(scores, label=label, context=CONTEXT, epsilon=2.0)["loss"]
    assert loss == pytest.approx(float(nll + 2.0 * (1 - probability).mean()), rel=1e-6)


def test_poly_loss_with_epsilon_zero_is_cross_entropy() -> None:
    scores, label = logits()
    plain = cross_entropy(scores, label=label, context=CONTEXT)["loss"]
    assert poly(scores, label=label, context=CONTEXT, epsilon=0.0)["loss"] == plain


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
