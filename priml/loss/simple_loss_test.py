"""Tests for SimpleLoss."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn
from torch.nn import functional

import pytest
import torch

from priml.cost import MEASURES, Cost, Kernel, Phase, cost
from priml.loss.custom_types import SimpleLossFn
from priml.loss.simple_loss import (
    SimpleLoss,
    bce_with_logits,
    cross_entropy,
    l1,
    mse,
)
from priml.loss.weighted_loss import WeightedSum
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from collections.abc import Mapping


def _fp32(
    *,
    primal: Mapping[str, Mapping[Kernel, int]] | None = None,
    adjoint: Mapping[str, Mapping[Kernel, int]] | None = None,
    **fields: int,
) -> Cost:
    """Build a ``Cost`` from whole-invocation fp32 FLOPs and bytes."""
    cells: dict[tuple[object, ...], int] = {}
    phases: tuple[tuple[Phase, Mapping[str, Mapping[Kernel, int]] | None], ...] = (
        ("primal", primal),
        ("adjoint", adjoint),
    )
    for phase, silos in phases:
        for measure in MEASURES:
            for kernel, value in (silos or {}).get(measure, {}).items():
                cells[(measure, phase, kernel, torch.float32)] = value
    return Cost(cells=cells, **fields)


class _StubLossFn:
    """Borrows the protocol's stub body, which computes nothing."""

    __call__ = SimpleLossFn.__call__


def test_simple_loss_fn_stub_computes_nothing() -> None:
    loss_fn: SimpleLossFn = _StubLossFn()
    assert loss_fn(torch.zeros(2), torch.zeros(2)) is None
    assert loss_fn(torch.zeros(2), torch.zeros(2), reduction="sum") is None


def test_simple_loss_basic():
    """Test SimpleLoss with default binary_cross_entropy_with_logits."""
    cfg = SimpleLoss.Config()
    loss = cfg.make()

    prediction = torch.randn(2, 3)
    batch = {"label": torch.randn(2, 3)}

    result = loss(prediction, **batch)

    assert "loss" in result
    assert result["loss"].shape == (2, 3)  # reduction="none"


def test_simple_loss_with_kwargs():
    """Test SimpleLoss with additional kwargs."""
    cfg = SimpleLoss.Config(
        kwargs={
            "reduction": "none",
            "pos_weight": torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0]),
        },
    )
    loss = cfg.make()

    prediction = torch.randn(4, 5)
    batch = {"label": torch.randn(4, 5)}

    result = loss(prediction, **batch)

    assert "loss" in result
    assert result["loss"].shape == (4, 5)


def test_simple_loss_custom_target_key():
    """Test SimpleLoss with custom target key."""
    cfg = SimpleLoss.Config(target_key="target")
    loss = cfg.make()

    prediction = torch.randn(3, 10)
    batch = {"target": torch.randn(3, 10)}

    result = loss(prediction, **batch)

    assert "loss" in result
    assert result["loss"].shape == (3, 10)


def test_simple_loss_missing_target():
    """Test SimpleLoss raises KeyError when target missing."""
    cfg = SimpleLoss.Config(target_key="label")
    loss = cfg.make()

    prediction = torch.randn(2, 3)
    batch = {"other": torch.tensor([0, 1])}

    with pytest.raises(KeyError) as exc_info:
        loss(prediction, **batch)
    assert "label" in str(exc_info.value)
    assert "Available keys" in str(exc_info.value)


def test_simple_loss_mse():
    """Test SimpleLoss with MSE loss function."""
    cfg = SimpleLoss.Config(
        loss_fn=functional.mse_loss,
        target_key="target",
    )
    loss = cfg.make()

    prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    batch = {"target": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

    result = loss(prediction, **batch)

    assert "loss" in result
    # Perfect match should have zero loss.
    assert torch.allclose(result["loss"], torch.zeros(2))


def test_simple_loss_reduction_mean():
    """Test SimpleLoss with reduction='mean' via kwargs."""
    cfg = SimpleLoss.Config(kwargs={"reduction": "mean"})
    loss = cfg.make()

    prediction = torch.randn(4, 5)
    batch = {"label": torch.randn(4, 5)}

    result = loss(prediction, **batch)

    assert "loss" in result
    assert result["loss"].ndim == 0  # Scalar.


def test_simple_loss_cost_bce_with_logits_counts_the_complete_invocation() -> None:
    """Default BCE: eight primal and five adjoint ops per logit, no reduction.

    The harness needs an ``nn.Module``, so the loss rides inside a one-term
    ``WeightedSum``; its two weighting ops are subtracted back out.
    """
    config = SimpleLoss.Config()
    label = torch.rand(4, 3)
    measured = assert_cost_matches_torch(
        WeightedSum.Config(fns=[config], weights=[1.0]),
        build_input=lambda: torch.randn(4, 3, requires_grad=True),
        seq_len=12,
        batch_size=1,
        dtype=None,
        run=lambda module, prediction: _loss(module, prediction, label=label),
    )
    expected = _fp32(
        primal={"flops": {"elementwise": 96}, "bytes": {"elementwise": 1008}},
        adjoint={"flops": {"elementwise": 60}, "bytes": {"elementwise": 480}},
    )
    assert cost(config, seq_len=12, batch_size=1, dtype=None) == expected
    assert measured == expected + _fp32(
        primal={
            "flops": {"elementwise": 24},
            "bytes": {"elementwise": 192, "reduction": 96},
        },
        adjoint={
            "flops": {"elementwise": 24},
            "bytes": {"elementwise": 96, "reduction": 96},
        },
    )
    assert measured.params == 0
    assert measured["flops", "matmul"].sum() == 0


def test_simple_loss_cost_cross_entropy_counts_complete_rows() -> None:
    """Cross-entropy costs a log-softmax over ``channels_out`` and one gather per row."""
    config = SimpleLoss.Config(
        loss_fn=cross_entropy,
        kwargs={"reduction": "mean"},
        channels_out=5,
    )
    label = torch.randint(0, 5, (4,))
    measured = assert_cost_matches_torch(
        WeightedSum.Config(fns=[config], weights=[1.0]),
        build_input=lambda: torch.randn(4, 5, requires_grad=True),
        seq_len=4,
        batch_size=1,
        dtype=None,
        run=lambda module, logits: _loss(module, logits, label=label),
    )
    # The label is one int64 read each way; the gathered logit and the
    # scattered ``-1`` are fp32 payload.
    label_bytes = Cost(
        cells={
            ("bytes", "primal", "selection", torch.int64): 32,
            ("bytes", "adjoint", "selection", torch.int64): 32,
        },
    )
    expected = _fp32(
        primal={
            "flops": {"elementwise": 68, "reduction": 35},
            "bytes": {"elementwise": 584, "reduction": 212, "selection": 32},
        },
        adjoint={
            "flops": {"elementwise": 44, "selection": 4},
            "bytes": {"elementwise": 368, "reduction": 20, "selection": 48},
        },
    )
    expected = expected + label_bytes
    assert cost(config, seq_len=4, batch_size=1, dtype=None) == expected
    assert measured == expected + _fp32(
        primal={
            "flops": {"elementwise": 8},
            "bytes": {"elementwise": 64, "reduction": 32},
        },
        adjoint={
            "flops": {"elementwise": 8},
            "bytes": {"elementwise": 32, "reduction": 32},
        },
    )
    assert measured["flops", "matmul"].sum() == 0


def test_simple_loss_cost_cross_entropy_needs_channels_out() -> None:
    config = SimpleLoss.Config(loss_fn=cross_entropy)
    with pytest.raises(ValueError, match="channels_out"):
        cost(config, seq_len=4, batch_size=1, dtype=None)


@pytest.mark.parametrize("loss_fn", [mse, l1])
def test_simple_loss_cost_regression_losses_are_two_ops(loss_fn: SimpleLossFn) -> None:
    """MSE and L1: subtract then square/abs; the adjoint scales the saved difference."""
    none = SimpleLoss.Config(loss_fn=loss_fn, kwargs={"reduction": "none"})
    expected = _fp32(
        primal={"flops": {"elementwise": 12}, "bytes": {"elementwise": 120}},
        adjoint={"flops": {"elementwise": 12}, "bytes": {"elementwise": 120}},
    )
    assert cost(none, seq_len=6, batch_size=1, dtype=None) == expected
    reduced = _fp32(
        primal={"flops": {"reduction": 5}, "bytes": {"reduction": 28}},
        adjoint={"bytes": {"reduction": 28}},
    )
    mean = SimpleLoss.Config(loss_fn=loss_fn, kwargs={"reduction": "mean"})
    assert cost(
        mean,
        seq_len=6,
        batch_size=1,
        dtype=None,
    ) == expected + reduced + _fp32(
        primal={"bytes": {"elementwise": 8}},
        adjoint={"flops": {"elementwise": 6}, "bytes": {"elementwise": 48}},
    )
    total = SimpleLoss.Config(loss_fn=loss_fn, kwargs={"reduction": "sum"})
    assert cost(total, seq_len=6, batch_size=1, dtype=None) == expected + reduced


def test_simple_loss_cost_rejects_unpriced_loss_fn() -> None:
    config = SimpleLoss.Config(loss_fn=functional.smooth_l1_loss)
    with pytest.raises(TypeError, match="smooth_l1_loss has no cost"):
        cost(config, seq_len=4, batch_size=1, dtype=None)


def test_the_stock_losses_are_torch_functionals_that_cost_themselves() -> None:
    x = torch.tensor([0.5, -1.5])
    y = torch.tensor([1.0, 0.0])
    torch.testing.assert_close(
        bce_with_logits(x, y),
        functional.binary_cross_entropy_with_logits(x, y),
    )
    torch.testing.assert_close(mse(x, y), functional.mse_loss(x, y))
    torch.testing.assert_close(l1(x, y), functional.l1_loss(x, y))
    logits = torch.tensor([[0.5, -1.5, 2.0]])
    label = torch.tensor([2])
    torch.testing.assert_close(
        cross_entropy(logits, label),
        functional.cross_entropy(logits, label),
    )
    costed = cost(mse, dtype=None, channels_out=-1, weighted=False, rescale=0)
    assert costed["flops", "primal", "elementwise", torch.float32] == 2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_simple_loss_operand_traffic(dtype: torch.dtype) -> None:
    config = SimpleLoss.Config()
    config.loss_fn = mse
    itemsize = dtype.itemsize
    costed = cost(config, seq_len=6, batch_size=1, dtype=dtype)
    assert costed["bytes", "primal", "elementwise"].sum() == 30 * itemsize
    assert costed["bytes", "adjoint", "elementwise"].sum() == 30 * itemsize
    config.kwargs = {"reduction": "sum"}
    reduced = cost(config, seq_len=6, batch_size=1, dtype=dtype)
    assert reduced["bytes", "primal", "reduction"].sum() == 7 * itemsize
    assert reduced["bytes", "adjoint", "reduction"].sum() == 7 * itemsize
    config.kwargs = {"reduction": "mean"}
    mean = cost(config, seq_len=6, batch_size=1, dtype=dtype)
    assert mean["bytes", "primal", "elementwise"].sum() == 32 * itemsize


def test_simple_loss_bce_weight_prices_multiply() -> None:
    config = SimpleLoss.Config()
    plain = cost(config, seq_len=6, batch_size=1, dtype=None)
    config.kwargs["weight"] = torch.ones(6)
    weighted = cost(config, seq_len=6, batch_size=1, dtype=None)
    assert weighted == plain + _fp32(
        primal={"flops": {"elementwise": 6}, "bytes": {"elementwise": 72}},
        adjoint={"flops": {"elementwise": 6}, "bytes": {"elementwise": 72}},
    )


@pytest.mark.parametrize(
    ("loss_fn", "option", "value"),
    [
        (bce_with_logits, "pos_weight", torch.ones(2)),
        (cross_entropy, "weight", torch.ones(2)),
        (cross_entropy, "label_smoothing", 0.1),
        (cross_entropy, "ignore_index", 0),
        (mse, "size_average", True),
        (l1, "reduce", False),
    ],
)
def test_simple_loss_rejects_unpriced_options(
    loss_fn: SimpleLossFn,
    option: str,
    value: object,
) -> None:
    config = SimpleLoss.Config(channels_out=2)
    config.loss_fn = loss_fn
    config.kwargs[option] = value
    with pytest.raises(NotImplementedError, match=option):
        cost(config, seq_len=6, batch_size=1, dtype=None)


@pytest.mark.parametrize(
    ("loss_fn", "options"),
    [
        (bce_with_logits, {"weight": None, "pos_weight": None}),
        (cross_entropy, {"weight": None, "label_smoothing": 0, "ignore_index": -100}),
        (mse, {"size_average": None, "reduce": None}),
    ],
)
def test_simple_loss_explicit_noop_options(
    loss_fn: SimpleLossFn,
    options: dict[str, object],
) -> None:
    config = SimpleLoss.Config(channels_out=2)
    config.loss_fn = loss_fn
    plain = cost(config, seq_len=6, batch_size=1, dtype=None)
    config.kwargs.update(options)
    assert cost(config, seq_len=6, batch_size=1, dtype=None) == plain


def _loss(module: nn.Module, prediction: Tensor, **batch: Tensor) -> Tensor:
    """Run the weighted-sum wrapper and return its ``loss`` tensor."""
    assert isinstance(module, WeightedSum)
    return module(prediction, **batch)["loss"]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
