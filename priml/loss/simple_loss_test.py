"""Tests for SimpleLoss."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn
from torch.nn import functional

import pytest
import torch

from priml.loss.custom_types import SimpleLossFn
from priml.loss.simple_loss import SimpleLoss
from priml.loss.weighted_loss import WeightedSum
from priml.model.cost import MEASURES, Cost, Kernel, Phase, cost
from priml.testing.cost import assert_cost_matches_torch


def _fp32(
    *,
    primal: Mapping[str, Mapping[Kernel, float]] | None = None,
    adjoint: Mapping[str, Mapping[Kernel, float]] | None = None,
    **fields: int,
) -> Cost:
    """Build a ``Cost`` from per-phase, per-kernel fp32 FLOPs and bytes."""
    cells: dict[tuple[object, ...], float] = {}
    phases: tuple[tuple[Phase, Mapping[str, Mapping[Kernel, float]] | None], ...] = (
        ("primal", primal),
        ("adjoint", adjoint),
    )
    for phase, silos in phases:
        for measure in MEASURES:
            for kernel, value in (silos or {}).get(measure, {}).items():
                cells[(measure, phase, kernel, torch.float32)] = value
    return Cost(cells=cells, **fields)


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


def test_simple_loss_cost_bce_with_logits_is_per_logit() -> None:
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
        primal={"flops": {"elementwise": 8}, "bytes": {"elementwise": 84}},
        adjoint={"flops": {"elementwise": 5}, "bytes": {"elementwise": 40}},
    )
    assert cost(config, seq_len=12, batch_size=1, dtype=None) == expected
    assert measured == expected + _fp32(
        primal={
            "flops": {"elementwise": 2},
            "bytes": {"elementwise": 16, "reduction": 8},
        },
        adjoint={
            "flops": {"elementwise": 2},
            "bytes": {"elementwise": 8, "reduction": 8},
        },
    )
    assert measured.params == 0
    assert measured["flops", :, "matmul"].sum() == 0


def test_simple_loss_cost_cross_entropy_is_per_row() -> None:
    """Cross-entropy prices a log-softmax over ``channels_out`` and one gather per row."""
    config = SimpleLoss.Config(
        loss_fn=functional.cross_entropy,
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
            ("bytes", "primal", "selection", torch.int64): 8,
            ("bytes", "adjoint", "selection", torch.int64): 8,
        },
    )
    expected = _fp32(
        primal={
            "flops": {"elementwise": 3 * 5 + 2, "reduction": 2 * (5 - 1) + (4 - 1) / 4},
            "bytes": {"elementwise": 144 + 2, "reduction": 48 + 5, "selection": 8},
        },
        adjoint={
            "flops": {"elementwise": 2 * 5 + 1, "selection": 1},
            "bytes": {"elementwise": 92, "reduction": 5, "selection": 12},
        },
    )
    expected = expected + label_bytes
    assert cost(config, seq_len=4, batch_size=1, dtype=None) == expected
    assert measured == expected + _fp32(
        primal={
            "flops": {"elementwise": 2},
            "bytes": {"elementwise": 16, "reduction": 8},
        },
        adjoint={
            "flops": {"elementwise": 2},
            "bytes": {"elementwise": 8, "reduction": 8},
        },
    )
    assert measured["flops", :, "matmul"].sum() == 0


def test_simple_loss_cost_cross_entropy_needs_channels_out() -> None:
    config = SimpleLoss.Config(loss_fn=functional.cross_entropy)
    with pytest.raises(ValueError, match="channels_out"):
        cost(config, seq_len=4, batch_size=1, dtype=None)


@pytest.mark.parametrize("loss_fn", [functional.mse_loss, functional.l1_loss])
def test_simple_loss_cost_regression_losses_are_two_ops(loss_fn: SimpleLossFn) -> None:
    """MSE and L1: subtract then square/abs; the adjoint scales the saved difference."""
    none = SimpleLoss.Config(loss_fn=loss_fn, kwargs={"reduction": "none"})
    expected = _fp32(
        primal={"flops": {"elementwise": 2}, "bytes": {"elementwise": 20}},
        adjoint={"flops": {"elementwise": 2}, "bytes": {"elementwise": 20}},
    )
    assert cost(none, seq_len=6, batch_size=1, dtype=None) == expected
    reduced = _fp32(
        primal={"flops": {"reduction": 5 / 6}, "bytes": {"reduction": 28 / 6}},
        adjoint={"bytes": {"reduction": 28 / 6}},
    )
    mean = SimpleLoss.Config(loss_fn=loss_fn, kwargs={"reduction": "mean"})
    assert cost(
        mean,
        seq_len=6,
        batch_size=1,
        dtype=None,
    ) == expected + reduced + _fp32(
        primal={"bytes": {"elementwise": 8 / 6}},
        adjoint={"flops": {"elementwise": 1}, "bytes": {"elementwise": 8}},
    )
    total = SimpleLoss.Config(loss_fn=loss_fn, kwargs={"reduction": "sum"})
    assert cost(total, seq_len=6, batch_size=1, dtype=None) == expected + reduced


def test_simple_loss_cost_rejects_unpriced_loss_fn() -> None:
    config = SimpleLoss.Config(loss_fn=functional.smooth_l1_loss)
    with pytest.raises(TypeError, match="smooth_l1_loss"):
        cost(config, seq_len=4, batch_size=1, dtype=None)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_simple_loss_operand_traffic(dtype: torch.dtype) -> None:
    config = SimpleLoss.Config()
    config.loss_fn = functional.mse_loss
    itemsize = dtype.itemsize
    priced = cost(config, seq_len=6, batch_size=1, dtype=dtype)
    assert priced["bytes", "primal", "elementwise"].sum() == 5 * itemsize
    assert priced["bytes", "adjoint", "elementwise"].sum() == 5 * itemsize
    config.kwargs = {"reduction": "sum"}
    reduced = cost(config, seq_len=6, batch_size=1, dtype=dtype)
    assert reduced["bytes", "primal", "reduction"].sum() == 7 * itemsize / 6
    assert reduced["bytes", "adjoint", "reduction"].sum() == 7 * itemsize / 6
    config.kwargs = {"reduction": "mean"}
    mean = cost(config, seq_len=6, batch_size=1, dtype=dtype)
    assert (
        mean["bytes", "primal", "elementwise"].sum() == 5 * itemsize + 2 * itemsize / 6
    )


def test_simple_loss_bce_weight_prices_multiply() -> None:
    config = SimpleLoss.Config()
    plain = cost(config, seq_len=6, batch_size=1, dtype=None)
    config.kwargs["weight"] = torch.ones(6)
    weighted = cost(config, seq_len=6, batch_size=1, dtype=None)
    assert weighted == plain + _fp32(
        primal={"flops": {"elementwise": 1}, "bytes": {"elementwise": 3 * 4}},
        adjoint={"flops": {"elementwise": 1}, "bytes": {"elementwise": 3 * 4}},
    )


@pytest.mark.parametrize(
    ("loss_fn", "option", "value"),
    [
        (functional.binary_cross_entropy_with_logits, "pos_weight", torch.ones(2)),
        (functional.cross_entropy, "weight", torch.ones(2)),
        (functional.cross_entropy, "label_smoothing", 0.1),
        (functional.cross_entropy, "ignore_index", 0),
        (functional.mse_loss, "size_average", True),
        (functional.l1_loss, "reduce", False),
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
        (
            functional.binary_cross_entropy_with_logits,
            {"weight": None, "pos_weight": None},
        ),
        (
            functional.cross_entropy,
            {"weight": None, "label_smoothing": 0, "ignore_index": -100},
        ),
        (functional.mse_loss, {"size_average": None, "reduce": None}),
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
