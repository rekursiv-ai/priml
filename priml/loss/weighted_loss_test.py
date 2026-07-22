"""Tests for WeightedSum."""

from __future__ import annotations

from typing import override

from configgle import Fig
from torch import nn

import torch

from priml.loss.simple_loss import SimpleLoss
from priml.loss.weighted_loss import WeightedSum


class DummyLoss(nn.Module):
    """Dummy loss that returns a constant value."""

    class Config(Fig["DummyLoss"]):
        value: float = 1.0

    def __init__(self, config: Config):
        super().__init__()
        self.value = config.value

    @override
    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return torch.tensor(self.value)


def test_weighted_sum_basic():
    """Test WeightedSum basic functionality."""
    cfg = WeightedSum.Config(
        fns=[DummyLoss.Config(value=1.0), DummyLoss.Config(value=2.0)],
        weights=[0.5, 0.5],
    )
    loss = cfg.make()

    result = loss()

    assert result["loss"].item() == 1.5  # 0.5*1.0 + 0.5*2.0
    assert "loss_0" in result
    assert "loss_1" in result
    assert result["loss_0"].item() == 1.0
    assert result["loss_1"].item() == 2.0


def test_weighted_sum_different_weights():
    """Test WeightedSum with different weights."""
    cfg = WeightedSum.Config(
        fns=[DummyLoss.Config(value=10.0), DummyLoss.Config(value=20.0)],
        weights=[0.3, 0.7],
    )
    loss = cfg.make()

    result = loss()

    assert result["loss"].item() == 17.0  # 0.3*10.0 + 0.7*20.0
    assert result["loss_0"].item() == 10.0
    assert result["loss_1"].item() == 20.0


def test_weighted_sum_single_loss():
    """Test WeightedSum with single loss."""
    cfg = WeightedSum.Config(
        fns=[DummyLoss.Config(value=5.0)],
        weights=[2.0],
    )
    loss = cfg.make()

    result = loss()

    assert result["loss"].item() == 10.0  # 2.0*5.0
    assert "loss_0" in result


def test_weighted_sum_three_losses():
    """Test WeightedSum with three losses."""
    cfg = WeightedSum.Config(
        fns=[
            DummyLoss.Config(value=1.0),
            DummyLoss.Config(value=2.0),
            DummyLoss.Config(value=3.0),
        ],
        weights=[1.0, 1.0, 1.0],
    )
    loss = cfg.make()

    result = loss()

    assert result["loss"].item() == 6.0  # 1.0*1.0 + 1.0*2.0 + 1.0*3.0
    assert "loss_0" in result
    assert "loss_1" in result
    assert "loss_2" in result


def test_weighted_sum_zero_weight():
    """Test WeightedSum with zero weight."""
    cfg = WeightedSum.Config(
        fns=[DummyLoss.Config(value=100.0), DummyLoss.Config(value=5.0)],
        weights=[0.0, 1.0],
    )
    loss = cfg.make()

    result = loss()

    assert result["loss"].item() == 5.0  # 0.0*100.0 + 1.0*5.0
    assert result["loss_0"].item() == 100.0  # Unscaled preserves original


def test_weighted_sum_accepts_plain_callable_loss() -> None:
    """LOSSOPT-011: composing non-nn.Module callables (SimpleLoss) must work."""
    cfg = WeightedSum.Config(
        fns=[SimpleLoss.Config()],
        weights=[1.0],
    )
    loss = cfg.make()

    prediction = torch.zeros(3)
    result = loss(prediction, label=torch.zeros(3))

    assert "loss" in result
    assert "loss_0" in result


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
