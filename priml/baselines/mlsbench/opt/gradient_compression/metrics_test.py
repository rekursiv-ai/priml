"""Tests for the mean test cross-entropy."""

from __future__ import annotations

from torch.nn import functional

import pytest
import torch

from priml.baselines.mlsbench.opt.gradient_compression.metrics import (
    MeanCrossEntropy,
)


def test_the_mean_is_over_images_not_batches() -> None:
    metric = MeanCrossEntropy.Config().make()
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, 3.0]])
    label = torch.tensor([0, 0, 1])
    metric.update(logits[:2], label=label[:2])
    metric.update(logits[2:], label=label[2:])
    # Each batch's sum leaves torch as a Python float before the two are added.
    first = functional.cross_entropy(logits[:2], label[:2], reduction="sum").item()
    last = functional.cross_entropy(logits[2:], label[2:], reduction="sum").item()
    assert metric.compute() == {"mean": (first + last) / 3}


def test_reset_and_state_round_trip() -> None:
    metric = MeanCrossEntropy.Config().make()
    metric.update(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), label=torch.tensor([0, 1]))
    restored = MeanCrossEntropy.Config().make()
    restored.load_state_dict(metric.state_dict())
    assert restored.compute() == metric.compute()
    metric.reset()
    with pytest.raises(ValueError, match="no examples"):
        metric.compute()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
