"""Tests for WeightedSum."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from configgle import Fig
from torch import Tensor, nn

import pytest
import torch

from priml.cost import Cost, cost
from priml.loss.simple_loss import SimpleLoss, mse
from priml.loss.weighted_loss import WeightedSum
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from priml.loss.custom_types import LossOutput


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
    """Children are costed through ``cost``; each adds a multiply and a stacked add."""
    bce = SimpleLoss.Config()
    regression = SimpleLoss.Config(loss_fn=mse)
    config = WeightedSum.Config(fns=[bce, regression], weights=[0.5, 0.5])
    label = torch.rand(4, 3)
    measured = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(4, 3, requires_grad=True),
        seq_len=12,
        batch_size=1,
        dtype=None,
        run=lambda module, prediction: _loss(module, prediction, label=label),
    )
    f32 = torch.float32
    assert measured == cost(bce, seq_len=12, batch_size=1, dtype=None) + cost(
        regression,
        seq_len=12,
        batch_size=1,
        dtype=None,
    ) + Cost(
        cells={
            ("flops", "primal", "elementwise", f32): 48,
            ("flops", "adjoint", "elementwise", f32): 48,
            ("bytes", "primal", "elementwise", f32): 384,
            ("bytes", "primal", "reduction", f32): 144,
            ("bytes", "adjoint", "elementwise", f32): 192,
            ("bytes", "adjoint", "reduction", f32): 144,
        },
    )
    assert measured.params == 0
    assert measured["flops", "matmul"].sum() == 0


def test_weighted_sum_cost_rejects_unpriced_child() -> None:
    """A child without ``cost`` raises instead of costing zero."""
    config = WeightedSum.Config(fns=[DummyLoss.Config()], weights=[1.0])
    with pytest.raises(TypeError, match=r"DummyLoss\.Config has no cost"):
        cost(config, seq_len=1, batch_size=1, dtype=None)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_weighted_sum_operand_traffic(dtype: torch.dtype) -> None:
    config = WeightedSum.Config()
    config.fns = [SimpleLoss.Config(), SimpleLoss.Config()]
    config.weights = [0.5, 0.5]
    itemsize = dtype.itemsize
    costed = cost(config, seq_len=6, batch_size=1, dtype=dtype)
    child = cost(config.fns[0], seq_len=6, batch_size=1, dtype=dtype)
    assert (
        costed["bytes", "primal", "elementwise"].sum()
        == 2 * child["bytes", "primal", "elementwise"].sum() + 48 * itemsize
    )
    assert costed["bytes", "primal", "reduction"].sum() == 18 * itemsize
    assert (
        costed["bytes", "adjoint", "elementwise"].sum()
        == 2 * child["bytes", "adjoint", "elementwise"].sum() + 24 * itemsize
    )
    assert costed["bytes", "adjoint", "reduction"].sum() == 18 * itemsize


def _loss(module: nn.Module, prediction: Tensor, **batch: Tensor) -> Tensor:
    """Run the weighted-sum wrapper and return its ``loss`` tensor."""
    assert isinstance(module, WeightedSum)
    return module(prediction, **batch)["loss"]


class _TensorLoss(nn.Module):
    """A loss that returns a bare tensor, the pre-dict contract."""

    class Config(Fig["_TensorLoss"]):
        pass

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()

    @override
    def forward(self, *args: object, **kwargs: object) -> Tensor:
        del kwargs
        prediction, target = args
        assert isinstance(prediction, Tensor)
        assert isinstance(target, Tensor)
        return (prediction - target).abs().mean()


def test_weighted_sum_accepts_a_bare_tensor_loss_and_names_it_by_index() -> None:
    config = WeightedSum.Config()
    config.fns = [_TensorLoss.Config(), _TensorLoss.Config()]
    config.weights = [1.0, 3.0]
    out = config.make()(torch.ones(2, 3), torch.zeros(2, 3))
    assert out["loss_0"] == 1
    assert out["loss_1"] == 1
    assert out["loss"] == 4.0


def test_weighted_sum_rejects_a_weight_count_that_differs_from_the_losses() -> None:
    config = WeightedSum.Config()
    config.fns = [_TensorLoss.Config()]
    config.weights = [1.0, 2.0]
    with pytest.raises(ValueError, match="count mismatch"):
        config.make()(torch.ones(2), torch.zeros(2))


class _DictLoss(nn.Module):
    """A loss that returns extra keys beside ``loss``."""

    class Config(Fig["_DictLoss"]):
        pass

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()

    @override
    def forward(self, *args: object, **kwargs: object) -> LossOutput:
        del kwargs
        prediction, target = args
        assert isinstance(prediction, Tensor)
        assert isinstance(target, Tensor)
        error = (prediction - target).abs()
        return {"loss": error.mean(), "max_error": error.max()}


def test_weighted_sum_suffixes_every_extra_key_with_the_loss_index() -> None:
    config = WeightedSum.Config()
    config.fns = [_DictLoss.Config(), _DictLoss.Config()]
    config.weights = [1.0, 1.0]
    out = config.make()(torch.tensor([1.0, 3.0]), torch.zeros(2))
    assert out["max_error_0"] == 3
    assert out["max_error_1"] == 3
    assert "max_error" not in out


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
