"""Tests for the class-weighted cross-entropy."""

from __future__ import annotations

from configgle import PartialConfig
from torch.nn import functional

import pytest
import torch

from priml.baselines.mlsbench.dl.vision.train_step import LossContext
from priml.baselines.mlsbench.dl.weighting.loss import (
    BindsClassCounts,
    ClassWeightedCrossEntropy,
)
from priml.baselines.mlsbench.dl.weighting.weights import inverse_frequency


CONTEXT = LossContext(num_classes=3, epoch=0, total_epochs=1)


def test_the_loss_weighs_each_example_by_its_class() -> None:
    config = ClassWeightedCrossEntropy.Config()
    config.class_weight = PartialConfig(inverse_frequency)
    loss = config.make()
    assert isinstance(loss, BindsClassCounts)
    counts = torch.tensor([4.0, 2.0, 1.0])
    loss.bind_class_counts(counts)
    logits = torch.arange(6.0).reshape(2, 3) / 4
    label = torch.tensor([0, 2])
    weight = inverse_frequency(counts, num_classes=3)
    assert torch.equal(
        loss(logits, label=label, context=CONTEXT)["loss"],
        functional.cross_entropy(logits, label, weight=weight),
    )


def test_uniform_weights_are_plain_cross_entropy() -> None:
    loss = ClassWeightedCrossEntropy.Config().make()
    loss.bind_class_counts(torch.tensor([4.0, 2.0, 1.0]))
    logits = torch.arange(6.0).reshape(2, 3) / 4
    label = torch.tensor([0, 2])
    assert torch.equal(
        loss(logits, label=label, context=CONTEXT)["loss"],
        functional.cross_entropy(logits, label),
    )


def test_the_loss_refuses_logits_of_another_class_count() -> None:
    loss = ClassWeightedCrossEntropy.Config().make()
    loss.bind_class_counts(torch.tensor([4.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="3 class weights but the logits have 2"):
        loss(torch.zeros(4, 2), label=torch.tensor([0, 1, 0, 1]), context=CONTEXT)


def test_the_weights_move_to_the_logits_device_once() -> None:
    loss = ClassWeightedCrossEntropy.Config().make()
    loss.bind_class_counts(torch.tensor([4.0, 2.0, 1.0]))
    logits = torch.zeros(2, 3, device="meta")
    _ = loss(
        logits,
        label=torch.zeros(2, dtype=torch.int64, device="meta"),
        context=CONTEXT,
    )
    moved = loss.weight
    assert moved is not None
    assert moved.device.type == "meta"
    _ = loss(
        logits,
        label=torch.zeros(2, dtype=torch.int64, device="meta"),
        context=CONTEXT,
    )
    assert loss.weight is moved


def test_the_loss_refuses_to_run_without_counts() -> None:
    loss = ClassWeightedCrossEntropy.Config().make()
    with pytest.raises(RuntimeError, match="bind_class_counts"):
        loss(torch.zeros(2, 3), label=torch.tensor([0, 1]), context=CONTEXT)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
