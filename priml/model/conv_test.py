"""Tests for conv module."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import math

from configgle.testing import assert_pprint_golden
from torch.utils.flop_counter import FlopCounterMode

import pytest
import torch

from priml.model.conv import Conv1d, Conv2d, Conv3d, conv_cost
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import (
    _convolution_backward_flops,
    _TrafficMode,
    assert_cost_matches_torch,
)


_CWD: Final = Path(__file__).resolve().parent


def test_conv1d_config_pprint() -> None:
    config = Conv1d.Config(2, 3)
    assert_pprint_golden(
        test_file=__file__,
        name="conv1d",
        config=config,
    )


def test_conv1d_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="conv1d",
        build_module=lambda: Conv1d.Config(2, 3).make(),
        build_input=lambda: torch.randn(1, 2, 4),
        seed=0,
    )


def test_conv2d_config_pprint() -> None:
    config = Conv2d.Config(2, 3)
    assert_pprint_golden(
        test_file=__file__,
        name="conv2d",
        config=config,
    )


def test_conv2d_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="conv2d",
        build_module=lambda: Conv2d.Config(2, 3).make(),
        build_input=lambda: torch.randn(1, 2, 3, 3),
        seed=0,
    )


def test_conv3d_config_pprint() -> None:
    config = Conv3d.Config(2, 3)
    assert_pprint_golden(
        test_file=__file__,
        name="conv3d",
        config=config,
    )


def test_conv3d_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="conv3d",
        build_module=lambda: Conv3d.Config(2, 3).make(),
        build_input=lambda: torch.randn(1, 2, 3, 3, 3),
        seed=0,
    )


def test_conv1d():
    m = Conv1d.Config(3, 16).make()
    x = torch.randn(2, 3, 32)
    assert m(x).shape == (2, 16, 32)


def test_conv1d_channels_infer():
    cfg = Conv1d.Config(channels_in=3).finalize()
    assert cfg.channels_out == 3


def test_conv2d():
    m = Conv2d.Config(3, 16, kernel_size=3, padding=1).make()
    x = torch.randn(2, 3, 32, 32)
    assert m(x).shape == (2, 16, 32, 32)


def test_conv2d_channels_infer():
    cfg = Conv2d.Config(channels_out=16).finalize()
    assert cfg.channels_in == 16


def test_conv3d():
    m = Conv3d.Config(3, 16, kernel_size=3, padding=1).make()
    x = torch.randn(2, 3, 8, 8, 8)
    assert m(x).shape == (2, 16, 8, 8, 8)


def test_conv3d_channels_infer():
    cfg = Conv3d.Config(channels_in=3).finalize()
    assert cfg.channels_out == 3


def test_conv_forward_accepts_messages_and_rejects_positional_extras():
    m = Conv2d.Config(3, 16, kernel_size=3, padding=1).make()
    x = torch.randn(2, 3, 32, 32)
    assert m(x, key="val").shape == (2, 16, 32, 32)
    with pytest.raises(TypeError):
        m(x, "extra")


def test_conv_reset():
    for cls in (Conv1d, Conv2d, Conv3d):
        m = cls.Config(3, 16).make()
        m.reset_parameters()


def test_conv2d_cost_is_a_matmul_over_the_receptive_field() -> None:
    """Count every output position: ``channels_in * prod(kernel_size)`` -> ``channels_out``.

    ``padding="same"`` keeps every input position as an output position, so
    a ``4 x 6`` image is 24 tokens.
    """
    analytical = assert_cost_matches_torch(
        Conv2d.Config(2, 3, kernel_size=(3, 5), bias=True),
        build_input=lambda: torch.randn(1, 2, 4, 6, requires_grad=True),
        input_grid=(4, 6),
        batch_size=1,
        dtype=None,
    )
    weights = 3 * 2 * 15
    assert analytical["flops", "primal", "matmul"].sum() == 2 * 24 * weights
    assert analytical["flops", "adjoint", "matmul"].sum() == 4 * 24 * weights
    assert analytical.params == weights + 3
    assert analytical["bytes", "primal", "elementwise"].sum() == 4 * 3


def test_conv1d_cost_divides_the_fan_in_by_groups() -> None:
    """Grouped convolution counts only its connected input channels in both passes."""
    analytical = assert_cost_matches_torch(
        Conv1d.Config(4, 6, kernel_size=3, groups=2),
        build_input=lambda: torch.randn(1, 4, 7, requires_grad=True),
        input_grid=(7,),
        batch_size=1,
        dtype=None,
    )
    weights = 6 * (4 // 2) * 3
    assert analytical["flops", "primal", "matmul"].sum() == 2 * 7 * weights
    assert analytical.params == weights
    assert analytical["bytes", "primal", "matmul"].sum() == 4 * (
        7 * 4 + 7 * 6 + weights
    )


def test_conv3d_cost_cubes_a_scalar_kernel() -> None:
    analytical = assert_cost_matches_torch(
        Conv3d.Config(2, 3, kernel_size=3),
        build_input=lambda: torch.randn(1, 2, 3, 4, 5, requires_grad=True),
        input_grid=(3, 4, 5),
        batch_size=1,
        dtype=None,
    )
    assert analytical["flops", "primal", "matmul"].sum() == 2 * (3 * 4 * 5) * (
        3 * 2 * 27
    )


@pytest.mark.parametrize(
    ("config_type", "ndim"),
    [(Conv1d.Config, 1), (Conv2d.Config, 2), (Conv3d.Config, 3)],
)
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("geometry", [(1, 0, 1), (2, 1, 2)])
def test_convolution_cost_tracks_input_and_output_grids(
    config_type: type[Conv1d.Config | Conv2d.Config | Conv3d.Config],
    ndim: int,
    groups: int,
    geometry: tuple[int, int, int],
) -> None:
    config = config_type()
    config.channels_in = 4
    config.channels_out = 6
    config.groups = groups
    config.stride, config.padding, config.dilation = geometry
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 4, *((7,) * ndim), requires_grad=True),
        input_grid=(7,) * ndim,
        batch_size=2,
        dtype=None,
    )


@pytest.mark.parametrize("gradient_mask", range(8))
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_convolution_cost_counts_only_requested_gradients(
    gradient_mask: int,
    groups: int,
    dtype: torch.dtype,
) -> None:
    input_grad, weight_grad, bias_grad = (
        bool(gradient_mask & (1 << index)) for index in range(3)
    )
    config = Conv2d.Config()
    config.channels_in = 4
    config.channels_out = 6
    config.groups = groups
    config.bias = True
    config.dtype = dtype
    config.padding = 1
    config.stride = 2
    module = config.make()
    module.weight.requires_grad_(weight_grad)
    assert module.bias is not None
    module.bias.requires_grad_(bias_grad)
    x = torch.randn(2, 4, 5, 7, dtype=dtype, requires_grad=input_grad)
    traffic = _TrafficMode()
    with (
        FlopCounterMode(
            display=False,
            custom_mapping={
                torch.ops.aten.convolution_backward: _convolution_backward_flops,
            },
        ) as counter,
        traffic,
    ):
        output = module(x)
        if output.requires_grad:
            output.sum().backward()
    rows = output.shape[0] * math.prod(output.shape[2:])
    analytical = conv_cost(
        channels_in=4,
        channels_out=6,
        kernel_size=3,
        ndim=2,
        groups=groups,
        bias=True,
        input_grid=(5, 7),
        batch_size=2,
        stride=2,
        padding=1,
        dtype=dtype,
        input_grad=input_grad,
        weight_grad=weight_grad,
        bias_grad=bias_grad,
    )
    weights = 6 * (4 // groups) * 9
    assert counter.get_total_flops() == 2 * weights * rows * (
        1 + int(input_grad) + int(weight_grad)
    )
    assert analytical["flops", "matmul"].sum() == counter.get_total_flops()
    assert analytical["bytes", "matmul"].sum() == traffic.bytes["matmul"]
    assert (x.grad is not None) == input_grad
    assert (module.weight.grad is not None) == weight_grad
    assert (module.bias.grad is not None) == bias_grad


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
