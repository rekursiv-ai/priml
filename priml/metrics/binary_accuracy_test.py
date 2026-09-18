"""Tests for BinaryAccuracy metric."""

from __future__ import annotations

import torch

from priml.metrics.binary_accuracy import BinaryAccuracy


def test_binary_accuracy_singleton_logit_dim() -> None:
    """LOSSOPT-008: [B,1] logits vs [B] targets must not broadcast to [B,B].

    The broadcast bug makes ``correct`` exceed ``total`` for unequal
    predictions; here correct counts 4 with total 3 before the squeeze.
    """
    metric = BinaryAccuracy(BinaryAccuracy.Config())
    logits = torch.tensor([[10.0], [10.0], [-10.0]])
    label = torch.tensor([1.0, 0.0, 0.0])

    metric.update(logits, label=label)
    metrics = metric.compute()

    assert metric.correct == 2
    assert metric.total == 3
    assert metric.correct <= metric.total
    assert metrics["accuracy"] == 2 / 3


def test_binary_accuracy_flat_logits() -> None:
    """Flat [B] logits remain correct after squeeze fix."""
    metric = BinaryAccuracy(BinaryAccuracy.Config())
    logits = torch.tensor([10.0, -10.0, 10.0])
    label = torch.tensor([1.0, 0.0, 0.0])

    metric.update(logits, label=label)

    assert metric.correct == 2
    assert metric.total == 3


def test_binary_accuracy_compute_is_zero_before_any_update() -> None:
    assert BinaryAccuracy(BinaryAccuracy.Config()).compute() == {"accuracy": 0.0}


def test_binary_accuracy_state_round_trips_and_reset_clears_it() -> None:
    metric = BinaryAccuracy(BinaryAccuracy.Config())
    metric.update(
        torch.tensor([10.0, -10.0, 10.0]),
        label=torch.tensor([1.0, 0.0, 0.0]),
    )
    assert metric.state_dict() == {"correct": 2, "total": 3}

    restored = BinaryAccuracy(BinaryAccuracy.Config())
    restored.load_state_dict(metric.state_dict())
    assert restored.compute() == {"accuracy": 2 / 3}

    metric.reset()
    assert metric.state_dict() == {"correct": 0, "total": 0}


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
