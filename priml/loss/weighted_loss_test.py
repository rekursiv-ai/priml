"""Tests for WeightedSum."""

from __future__ import annotations

from typing import override

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import pytest
import torch

from priml.loss.simple_loss import SimpleLoss
from priml.loss.weighted_loss import WeightedSum
from priml.model.cost import Bytes, Compute, Cost, Flops, cost
from priml.testing.cost import assert_cost_matches_torch


class DummyLoss(nn.Module):
    """Dummy loss that returns a constant value."""

    class Config(Fig["DummyLoss"]):
        value: float = 1.0

    def __init__(self, config: Config):
        super().__init__()
        self.value = config.value

    @override
    def forward(self, *args: object, **kwargs: object) -> Tensor:
        return torch.tensor(self.value)


def test_weighted_sum_basic():
    """Test WeightedSum basic functionality."""
    cfg = WeightedSum.Config(
        fns=[DummyLoss.Config(value=1.0), DummyLoss.Config(value=2.0)],
        weights=[0.5, 0.5],
    )
    loss = cfg.make()

    result = loss()

    assert result["loss"].item() == 1.5  # 0.5*1.0 + 0.5*2.0.
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

    assert result["loss"].item() == 17.0  # 0.3*10.0 + 0.7*20.0.
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

    assert result["loss"].item() == 10.0  # 2.0*5.0.
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

    assert result["loss"].item() == 6.0  # 1.0*1.0 + 1.0*2.0 + 1.0*3.0.
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

    assert result["loss"].item() == 5.0  # 0.0*100.0 + 1.0*5.0.
    assert result["loss_0"].item() == 100.0  # Unscaled preserves original.


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


def test_weighted_sum_cost_sums_children_plus_weighting() -> None:
    """Children are priced through ``cost``; each adds a multiply and a stacked add."""
    bce = SimpleLoss.Config()
    mse = SimpleLoss.Config(loss_fn=functional.mse_loss)
    config = WeightedSum.Config(fns=[bce, mse], weights=[0.5, 0.5])
    label = torch.rand(4, 3)
    measured = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(4, 3, requires_grad=True),
        num_tokens=12,
        run=lambda module, prediction: _loss(module, prediction, label=label),
    )
    assert measured == cost(bce, rows=12) + cost(
        mse,
        rows=12,
    ) + Cost(
        primal=Compute(
            flops=Flops(elementwise=4),
            bytes=Bytes(elementwise=32, reduction=12),
        ),
        adjoint=Compute(
            flops=Flops(elementwise=4),
            bytes=Bytes(elementwise=16, reduction=12),
        ),
    )
    assert measured.params == 0
    assert measured.training.flops.matmul == 0


def test_weighted_sum_cost_rejects_unpriced_child() -> None:
    """A child without ``cost`` raises instead of pricing zero."""
    config = WeightedSum.Config(fns=[DummyLoss.Config()], weights=[1.0])
    with pytest.raises(TypeError, match=r"DummyLoss\.Config has no cost"):
        cost(config, rows=1)


@pytest.mark.parametrize("itemsize", [2, 4, 8])
def test_weighted_sum_operand_traffic(itemsize: int) -> None:
    config = WeightedSum.Config()
    config.fns = [SimpleLoss.Config(), SimpleLoss.Config()]
    config.weights = [0.5, 0.5]
    priced = cost(config, rows=6, itemsize=itemsize)
    child = cost(config.fns[0], rows=6, itemsize=itemsize)
    assert (
        priced.primal.bytes.elementwise
        == 2 * child.primal.bytes.elementwise + 8 * itemsize
    )
    assert priced.primal.bytes.reduction == 3 * itemsize
    assert (
        priced.adjoint.bytes.elementwise
        == 2 * child.adjoint.bytes.elementwise + 4 * itemsize
    )
    assert priced.adjoint.bytes.reduction == 3 * itemsize


def _loss(module: nn.Module, prediction: Tensor, **batch: Tensor) -> Tensor:
    """Run the weighted-sum wrapper and return its ``loss`` tensor."""
    assert isinstance(module, WeightedSum)
    return module(prediction, **batch)["loss"]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
