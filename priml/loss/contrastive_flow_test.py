"""Tests for contrastive flow matching."""

from __future__ import annotations

import torch

from priml.loss.contrastive_flow import contrastive_flow_loss


def test_loss_rolls_targets_and_applies_broadcast_weight() -> None:
    prediction = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    target = torch.tensor([[1.0, 2.0], [4.0, 6.0]])
    weight = torch.tensor([[2.0], [3.0]])
    expected = -(((prediction - torch.roll(target, 1, 0)) ** 2) * weight).mean()
    assert contrastive_flow_loss(prediction, target, weight=weight) == expected


def test_loss_without_weight_is_negative_mean_squared_error() -> None:
    prediction = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    target = torch.tensor([[1.0, 2.0], [4.0, 6.0]])
    expected = -((prediction - torch.roll(target, 1, 0)) ** 2).mean()
    assert contrastive_flow_loss(prediction, target) == expected


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
