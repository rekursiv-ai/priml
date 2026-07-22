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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
