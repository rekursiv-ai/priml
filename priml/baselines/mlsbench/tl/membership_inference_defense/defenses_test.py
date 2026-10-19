"""Tests for the published defenses, against their closed forms."""

from __future__ import annotations

from torch.nn import functional

import pytest
import torch

from priml.baselines.mlsbench.dl.vision.train_step import LossContext
from priml.baselines.mlsbench.tl.membership_inference_defense.defenses import (
    confidence_penalty,
    relaxloss,
)
from priml.baselines.mlsbench.tl.membership_inference_defense.reference import (
    defense_cases,
)


def context(epoch: int, classes: int = 4) -> LossContext:
    """Describe an epoch of a run to a defense."""
    return LossContext(num_classes=classes, epoch=epoch, total_epochs=300)


def test_the_confidence_penalty_subtracts_the_weighted_entropy() -> None:
    logits, label = defense_cases()["low"]
    probs = functional.softmax(logits, dim=1)
    entropy = -(probs * probs.log()).sum(dim=1).mean()
    loss = confidence_penalty(
        logits,
        label=label,
        context=context(0),
        entropy_weight=0.25,
        min_probability=1e-8,
    )["loss"]
    expected = functional.cross_entropy(logits, label) - 0.25 * entropy
    torch.testing.assert_close(loss, expected)


def test_the_confidence_penalty_floors_the_logarithm() -> None:
    logits, label = defense_cases()["wide"]
    loss = confidence_penalty(
        logits,
        label=label,
        context=context(0, 100),
        entropy_weight=1.0,
        min_probability=1e-8,
    )["loss"]
    assert torch.isfinite(loss)


@pytest.mark.parametrize("case", ["low", "high"])
def test_relaxloss_holds_the_loss_at_its_level_in_an_even_epoch(case: str) -> None:
    logits, label = defense_cases()[case]
    loss = relaxloss(logits, label=label, context=context(2), alpha=1.0, upper=1.0)
    ce = functional.cross_entropy(logits, label)
    torch.testing.assert_close(loss["loss"], (ce - 1.0).abs())


def test_relaxloss_descends_above_its_level_in_an_odd_epoch() -> None:
    logits, label = defense_cases()["high"]
    loss = relaxloss(logits, label=label, context=context(1), alpha=1.0, upper=1.0)
    torch.testing.assert_close(loss["loss"], functional.cross_entropy(logits, label))


def test_relaxloss_flattens_only_the_wrong_examples_below_its_level() -> None:
    logits, label = defense_cases()["low"]
    loss = relaxloss(logits, label=label, context=context(1), alpha=1.0, upper=1.0)
    probs = functional.softmax(logits, dim=1)
    kept = probs[torch.arange(3), label]
    target = ((1 - kept) / 3)[:, None].expand(3, 4).clone()
    target[torch.arange(3), label] = kept
    flattened = -(target * functional.log_softmax(logits, dim=1)).sum(dim=1)
    ce = functional.cross_entropy(logits, label, reduction="none")
    wrong = torch.tensor([0.0, 0.0, 1.0])
    torch.testing.assert_close(loss["loss"], (wrong * flattened - ce).mean())


def test_relaxloss_backpropagates_through_the_logits_only() -> None:
    logits, label = defense_cases()["low"]
    leaf = logits.clone().requires_grad_()
    relaxloss(leaf, label=label, context=context(1), alpha=1.0, upper=1.0)[
        "loss"
    ].backward()
    assert leaf.grad is not None
    # A correct example only ascends its own cross-entropy: gradient
    # -(onehot - softmax) / B, with the target detached.
    probs = functional.softmax(logits, dim=1)
    expected = (probs[0] - functional.one_hot(label[0], 4)) * -1 / 3
    torch.testing.assert_close(leaf.grad[0], expected)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
