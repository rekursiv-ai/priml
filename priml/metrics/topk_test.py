"""Tests for TopK accuracy metric."""

from __future__ import annotations

import pytest
import torch

from priml.metrics.topk import TopK


def test_topk_k_exceeds_num_classes() -> None:
    """LOSSOPT-006: default k=[1,5] must not crash on 3-class logits."""
    metric = TopK(TopK.Config())
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    label = torch.tensor([0, 2])

    metric.update(logits, label=label)
    metrics = metric.compute()

    assert metrics["top1"] == 1.0
    # With only 3 classes, top5 accuracy is always perfect.
    assert metrics["top5"] == 1.0


def test_topk_empty_k_values_raises_at_construction() -> None:
    """LOSSOPT-007: empty k_values must raise a clear ValueError early."""
    with pytest.raises(ValueError, match="k_values"):
        TopK(TopK.Config(k_values=[]))


def test_topk_update_requires_a_tensor_label() -> None:
    metric = TopK(TopK.Config(k_values=[1]))
    logits = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="label must be provided"):
        metric.update(logits)
    with pytest.raises(TypeError, match="label must be a Tensor"):
        metric.update(logits, label=[0, 1])


def test_topk_compute_is_zero_before_any_update() -> None:
    metric = TopK(TopK.Config(k_values=[1, 2]))
    assert metric.compute() == {"top1": 0.0, "top2": 0.0}


def test_topk_state_round_trips_and_reset_clears_it() -> None:
    metric = TopK(TopK.Config(k_values=[1, 2]))
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 2.0, 0.0]])
    metric.update(logits, label=torch.tensor([0, 2, 0]))
    assert metric.state_dict() == {"correct": {1: 1, 2: 3}, "total": 3}

    restored = TopK(TopK.Config(k_values=[1, 2]))
    restored.load_state_dict(metric.state_dict())
    assert restored.compute() == {"top1": 1 / 3, "top2": 1.0}
    # The restored counts are a copy, so the source's later updates stay out.
    metric.update(logits[:1], label=torch.tensor([0]))
    assert metric.state_dict() == {"correct": {1: 2, 2: 4}, "total": 4}
    assert restored.state_dict() == {"correct": {1: 1, 2: 3}, "total": 3}

    metric.reset()
    assert metric.state_dict() == {"correct": {1: 0, 2: 0}, "total": 0}


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
