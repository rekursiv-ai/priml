"""Tests for the ETTh1 DLinear model."""

from pathlib import Path
from typing import Final, cast

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.etth1.model import MovingAverage, SeriesDecomposition
from priml.baselines.etth1.testing import model_record, tiny_config
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import mismatches, read_tensors


_GOLDEN: Final = Path(__file__).parent / "testdata" / "dlinear_model.pt"


def test_model_matches_source_golden() -> None:
    with host_agnostic_numerics():
        torch.manual_seed(2021)
        actual = model_record(tiny_config().model.make())
    assert not mismatches(read_tensors(_GOLDEN), actual)


def test_model_golden_detects_one_ulp_change() -> None:
    expected = read_tensors(_GOLDEN)
    with host_agnostic_numerics():
        torch.manual_seed(2021)
        model = tiny_config().model.make()
    # Change one float32 ULP before recording the model.
    with torch.no_grad():
        value = model.seasonal.weight
        value[0, 0] = torch.nextafter(value[0, 0], torch.tensor(float("inf")))
    with host_agnostic_numerics():
        actual = model_record(model)
    assert "initial/seasonal.weight: 1/15 differ" in mismatches(expected, actual)


@pytest.mark.parametrize("kernel_size", [0, 2, -3])
def test_moving_average_rejects_invalid_width(kernel_size: int) -> None:
    cfg = MovingAverage.Config()
    cfg.kernel_size = kernel_size
    with pytest.raises(ValueError, match="positive and odd"):
        cfg.make()


def test_endpoint_padding_and_gradient() -> None:
    cfg = MovingAverage.Config()
    cfg.kernel_size = 3
    x = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
    x.requires_grad_()
    expected = torch.stack(
        [
            (2 * x[:, 0] + x[:, 1]) / 3,
            (x[:, 0] + x[:, 1] + x[:, 2]) / 3,
            (x[:, 1] + x[:, 2] + x[:, 3]) / 3,
            (x[:, 2] + x[:, 3] + x[:, 4]) / 3,
            (x[:, 3] + 2 * x[:, 4]) / 3,
        ],
        dim=1,
    )
    actual = cfg.make()(x)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual.sum(), x)[0], torch.ones_like(x))


def test_cost_counts_actual_projections_and_inactive_decoder() -> None:
    cfg = tiny_config().model
    result = assert_cost_matches_torch(
        cfg,
        build_input=lambda: torch.randn(2, 5, 4, requires_grad=True),
        seq_len=5,
        batch_size=2,
        dtype=torch.float32,
    )
    assert result.params == 3 * (5 + 1) * 3
    assert result.params_active == 2 * (5 + 1) * 3
    assert result["flops", "primal", "matmul"].sum() == 2 * 2 * (2 * 4) * 5 * 3
    with pytest.raises(ValueError, match="sequence length"):
        cfg.cost(seq_len=6, batch_size=2, dtype=torch.float32)


def _sum_decomposition(model: nn.Module, x: Tensor) -> Tensor:
    seasonal, trend = cast(tuple[Tensor, Tensor], model(x))
    return seasonal + trend


def test_decomposition_cost_has_no_parameters_or_matmuls() -> None:
    result = assert_cost_matches_torch(
        SeriesDecomposition.Config(),
        build_input=lambda: torch.randn(2, 5, 4, requires_grad=True),
        run=_sum_decomposition,
        seq_len=5,
        batch_size=2,
        channels=4,
        dtype=torch.float32,
    )
    assert result.params == 0
    assert result["flops", "primal"].sum() == 2 * 5 * 4 * (25 + 1)


def test_model_rejects_empty_history() -> None:
    cfg = tiny_config().model
    cfg.seq_len = 0
    with pytest.raises(ValueError, match="must be positive"):
        cfg.make()
