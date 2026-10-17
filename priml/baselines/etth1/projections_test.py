"""Check the compact projection's math, gradients, and cost."""

from typing import cast

import warnings

from torch import nn

import pytest
import torch

from priml.baselines.etth1.model import ReferenceLinear
from priml.baselines.etth1.projections import LowRankProjection
from priml.baselines.etth1.testing import tiny_config
from priml.testing.cost import assert_cost_matches_torch


def test_projection_starts_at_the_window_mean() -> None:
    model = LowRankProjection.Config(5, 3, channels_hidden=2).make()
    x = torch.randn(2, 4, 5)
    actual = model(x)
    assert actual.shape == (2, 4, 3)
    expected = x.sum(dim=-1) / 5
    for horizon in range(3):
        torch.testing.assert_close(actual[:, :, horizon], expected)


def test_projection_matches_composed_weights_and_gradients() -> None:
    model = LowRankProjection.Config(5, 3, channels_hidden=2).make().double()
    down = cast(nn.Linear, model.proj_in)
    up = cast(nn.Linear, model.proj_out)
    assert up.bias is not None
    with torch.no_grad():
        up.weight.normal_()
    x = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    actual = model(x)
    combined = up.weight @ down.weight + 1 / 5
    expected = x @ combined.T + up.bias
    torch.testing.assert_close(actual, expected)
    inputs = (x, down.weight, up.weight, up.bias)
    actual_grad = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), inputs)
    for actual_value, expected_value in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual_value, expected_value)


def test_injected_layers_own_their_weights_and_costs() -> None:
    cfg = LowRankProjection.Config(5, 3, channels_hidden=2)
    cfg.proj_in = ReferenceLinear.Config(bias=True)
    cfg.proj_out = ReferenceLinear.Config(bias=False)
    model = cfg.make()
    assert isinstance(model.proj_in, ReferenceLinear)
    assert isinstance(model.proj_out, ReferenceLinear)
    assert model.proj_in.bias is not None
    assert model.proj_out.bias is None
    assert (model.proj_in.in_features, model.proj_in.out_features) == (5, 2)
    assert (model.proj_out.in_features, model.proj_out.out_features) == (2, 3)
    result = assert_cost_matches_torch(
        cfg,
        build_input=lambda: torch.randn(2, 4, 5, requires_grad=True),
        seq_len=4,
        batch_size=2,
        dtype=torch.float32,
    )
    assert result.params == 5 * 2 + 2 + 2 * 3


def test_compact_model_cost_matches_its_training_graph() -> None:
    cfg = tiny_config().model
    cfg.seasonal = LowRankProjection.Config(channels_hidden=2)
    cfg.trend = LowRankProjection.Config(channels_hidden=2)
    result = assert_cost_matches_torch(
        cfg,
        build_input=lambda: torch.randn(2, 5, 4, requires_grad=True),
        seq_len=5,
        batch_size=2,
        dtype=torch.float32,
    )
    assert result.params_active == 2 * (5 * 2 + 2 * 3 + 3)
    assert result.params == result.params_active + (5 + 1) * 3
    assert result["flops", "primal", "matmul"].sum() == 2 * 2 * 2 * 4 * (5 * 2 + 2 * 3)


@pytest.mark.parametrize(
    ("history", "horizon", "hidden"),
    [(0, 3, 2), (5, 0, 2), (5, 3, 0)],
)
def test_projection_rejects_empty_widths(
    history: int,
    horizon: int,
    hidden: int,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        LowRankProjection.Config(history, horizon, channels_hidden=hidden).make()


def test_unbound_projection_prints_without_finalize_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rendered = LowRankProjection.Config().pformat(hide_default_values=False)
    assert "channels_hidden=32" in rendered


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
