"""Tests for conv module."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import math

from torch.utils.flop_counter import FlopCounterMode

import pytest
import torch

from priml.cost import cost, map_cost, set_cost
from priml.model.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    SequenceConv1d,
    conv_cost,
    conv_output_grid,
)
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import (
    _convolution_backward_flops,
    _TrafficMode,
    assert_cost_matches_torch,
)
from priml.testing.golden import assert_pprint_golden


_CWD: Final = Path(__file__).resolve().parent


@set_cost(map_cost(primal=1, adjoint=1))
def _costed_relu(value: torch.Tensor) -> torch.Tensor:
    """Apply ReLU with its one-operation-per-element cost declaration."""
    return torch.relu(value)


def test_conv1d_config_pprint() -> None:
    config = Conv1d.Config(2, channels_out=3)
    assert_pprint_golden(
        test_file=__file__,
        name="conv1d",
        config=config,
    )


def test_conv1d_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="conv1d",
        build_module=lambda: Conv1d.Config(2, channels_out=3).make(),
        build_input=lambda: torch.randn(3, 2, 4),
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
        build_input=lambda: torch.randn(3, 2, 4, 5),
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
        build_input=lambda: torch.randn(3, 2, 4, 5, 6),
        seed=0,
    )


@pytest.mark.parametrize(
    ("config", "grid"),
    [
        (Conv1d.Config(2, 3, kernel_size=2, stride=2, padding=0), (5,)),
        (Conv2d.Config(2, 3, kernel_size=2, stride=2, padding=0), (5, 6)),
        (Conv3d.Config(2, 3, kernel_size=2, stride=2, padding=0), (5, 6, 7)),
    ],
)
def test_declared_output_grid_matches_convolution(
    config: Conv1d.Config | Conv2d.Config | Conv3d.Config,
    grid: tuple[int, ...],
) -> None:
    layer = config.make()
    assert layer(torch.randn(4, 2, *grid)).shape[2:] == config.output_grid(grid)


def test_conv1d():
    m = Conv1d.Config(3, channels_out=16).make()
    x = torch.randn(2, 3, 32)
    assert m(x).shape == (2, 16, 32)


@pytest.mark.parametrize(
    ("kernel_size", "stride", "padding", "dilation"),
    [
        ((2,), 1, 0, 1),
        (2, (1,), 0, 1),
        (2, 1, (0,), 1),
        (2, 1, 0, (1,)),
    ],
)
def test_conv_output_grid_rejects_incomplete_axis_tuples(
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...],
    padding: int | tuple[int, ...],
    dilation: int | tuple[int, ...],
) -> None:
    """Reject per-axis tuple geometry that omits an input axis."""
    with pytest.raises(ValueError, match=r".+"):
        conv_output_grid(
            (7, 11),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )


def test_conv_output_grid_applies_tuple_geometry_per_axis() -> None:
    """Apply tuple geometry with floor((size + 2*pad - dil*(kernel-1)-1)/stride)+1.

    These axes yield outputs 3 and 9.
    """
    output_grid = conv_output_grid(
        (7, 11),
        kernel_size=(2, 3),
        stride=(2, 1),
        padding=(0, 1),
        dilation=(1, 2),
    )
    assert output_grid == (3, 9)


def test_sequence_conv1d_crops_to_input_sequence_length() -> None:
    mixer = SequenceConv1d.Config(
        conv=Conv1d.Config(3, channels_out=3, kernel_size=2, padding=1, bias=True),
    ).make()
    inputs = torch.randn(2, 5, 3)

    output = mixer(inputs)

    expected = torch.nn.functional.silu(
        mixer.conv(inputs.transpose(-1, -2))[..., : inputs.shape[-2]],
    ).transpose(-1, -2)
    assert output.shape == inputs.shape
    assert torch.equal(output, expected)


def test_sequence_conv1d_rejects_a_convolution_shorter_than_the_input() -> None:
    config = SequenceConv1d.Config(
        conv=Conv1d.Config(
            3,
            channels_out=4,
            kernel_size=3,
            padding="valid",
        ),
    )
    module = config.make()
    inputs = torch.randn(2, 4, 3)

    with pytest.raises(ValueError, match="output length 2"):
        module(inputs)


def test_sequence_conv1d_cost_rejects_a_convolution_shorter_than_the_input() -> None:
    config = SequenceConv1d.Config(
        conv=Conv1d.Config(
            3,
            channels_out=4,
            kernel_size=3,
            padding="valid",
        ),
    )

    with pytest.raises(ValueError, match="output length 2"):
        config.cost(seq_len=4, batch_size=2, dtype=None)


def test_sequence_conv_accepts_an_unbatched_sequence_and_resets_its_child() -> None:
    config = SequenceConv1d.Config(
        conv=Conv1d.Config(3, channels_out=4, kernel_size=1, bias=True),
    )
    module = config.make()
    torch.manual_seed(31)
    module.reset_parameters()
    torch.manual_seed(31)
    expected = config.make()
    assert torch.equal(module.conv.weight, expected.conv.weight)
    assert module.conv.bias is not None
    assert expected.conv.bias is not None
    assert torch.equal(module.conv.bias, expected.conv.bias)
    sequence = torch.randn(7, 3)
    assert module(sequence).shape == (7, 4)


def test_sequence_conv_flattens_and_restores_arbitrary_leading_dimensions() -> None:
    """Each leading index is an independent Conv1d batch row."""
    module = SequenceConv1d.Config(
        conv=Conv1d.Config(5, channels_out=4, kernel_size=3, padding=1, bias=True),
    ).make()
    sequence = torch.randn(2, 3, 7, 5, requires_grad=True)
    flattened = sequence.reshape(6, 7, 5)

    actual = module(sequence)
    expected = module(flattened).reshape(2, 3, 7, 4)
    assert actual.shape == (2, 3, 7, 4)
    assert torch.equal(actual, expected)

    actual.sum().backward()
    actual_gradient = sequence.grad
    assert actual_gradient is not None
    sequence.grad = None
    expected.sum().backward()
    assert sequence.grad is not None
    assert torch.equal(actual_gradient, sequence.grad)


def test_sequence_conv1d_casts_to_child_dtype_and_restores_input_dtype() -> None:
    mixer = SequenceConv1d.Config(
        conv=Conv1d.Config(3, channels_out=3, kernel_size=2, padding=1, bias=True),
    ).make()
    inputs = torch.randn(2, 5, 3, dtype=torch.bfloat16)

    output = mixer(inputs)

    expected = (
        torch.nn.functional.silu(
            mixer.conv(inputs.float().transpose(-1, -2))[..., : inputs.shape[-2]],
        )
        .transpose(-1, -2)
        .to(inputs.dtype)
    )
    assert output.dtype is inputs.dtype
    assert torch.equal(output, expected)


def test_sequence_conv1d_config_has_a_default_and_costs_the_crop() -> None:
    config = SequenceConv1d.Config()
    config.conv.channels_in = 3
    config.conv.channels_out = 4
    config.conv.kernel_size = 2
    config.conv.padding = 1
    config.conv.bias = True
    config.conv.init_bias = torch.nn.init.zeros_

    cost = config.finalize().cost(seq_len=5, batch_size=2, dtype=None)

    assert cost.params == 4 * 3 * 2 + 4
    assert cost["flops", "primal", "matmul"].sum() == 2 * 2 * 4 * 3 * 2 * 6
    assert cost["flops", "primal", "elementwise"].sum() == 5 * 5 * 2 * 4 + 6 * 2 * 4


def test_sequence_conv1d_config_adopts_explicit_child_widths() -> None:
    """Adopt explicit child widths into unset wrapper fields."""
    config = SequenceConv1d.Config()
    config.conv.channels_in = 3
    config.conv.channels_out = 4

    config.finalize()

    assert (config.channels_in, config.channels_out) == (3, 4)
    assert (config.conv.channels_in, config.conv.channels_out) == (3, 4)


def test_sequence_conv1d_rejects_conflicting_wrapper_and_child_widths() -> None:
    """Reject explicit child widths that disagree with the wrapper."""
    config = SequenceConv1d.Config(
        channels_in=3,
        channels_out=4,
        conv=Conv1d.Config(3, channels_out=5, kernel_size=1),
    )

    with pytest.raises(ValueError, match="channels_out"):
        config.make()


def test_sequence_conv1d_costs_an_injected_activation() -> None:
    """Non-default activations use their declared elementwise cost."""
    config = SequenceConv1d.Config(
        conv=Conv1d.Config(2, channels_out=3, kernel_size=1),
        act=_costed_relu,
    )
    measured = cost(config, seq_len=4, batch_size=2, dtype=torch.float32)
    convolution = cost(config.conv, input_grid=4, batch_size=2, dtype=torch.float32)
    activation = cost(_costed_relu, channels=24, dtype=torch.float32)

    assert measured == convolution + activation


def test_conv1d_channels_infer():
    cfg = Conv1d.Config(channels_in=3).finalize()
    assert cfg.channels_out == 3


def test_conv1d_depthwise_group_sentinel_uses_finalized_width() -> None:
    config = Conv1d.Config(channels_out=4, groups=-1).finalize()
    assert config.channels_in == 4
    assert config.groups == 4


def test_conv1d_depthwise_group_sentinel_requires_matching_widths() -> None:
    with pytest.raises(ValueError, match="Depthwise groups require matching"):
        Conv1d.Config(channels_in=3, channels_out=4, groups=-1).finalize()


def test_conv2d():
    m = Conv2d.Config(3, 16, kernel_size=3, padding=1).make()
    x = torch.randn(2, 3, 8, 12)
    assert m(x).shape == (2, 16, 8, 12)


def test_conv2d_channels_infer():
    cfg = Conv2d.Config(channels_out=16).finalize()
    assert cfg.channels_in == 16


def test_conv3d():
    m = Conv3d.Config(3, 16, kernel_size=3, padding=1).make()
    x = torch.randn(2, 3, 4, 5, 6)
    assert m(x).shape == (2, 16, 4, 5, 6)


def test_conv3d_channels_infer():
    cfg = Conv3d.Config(channels_in=3).finalize()
    assert cfg.channels_out == 3


def test_conv_forward_accepts_messages_and_rejects_positional_extras():
    m = Conv2d.Config(3, 16, kernel_size=3, padding=1).make()
    x = torch.randn(2, 3, 8, 12)
    assert m(x, key="val").shape == (2, 16, 8, 12)
    with pytest.raises(TypeError):
        m(x, "extra")


def test_conv_reset():
    for cls in (Conv1d, Conv2d, Conv3d):
        m = cls.Config(3, 16).make()
        m.reset_parameters()


@pytest.mark.parametrize(
    ("config_type", "given", "inferred"),
    [
        (Conv1d.Config, "channels_out", "channels_in"),
        (Conv2d.Config, "channels_in", "channels_out"),
        (Conv3d.Config, "channels_out", "channels_in"),
    ],
)
def test_conv_infers_either_width_from_the_other(
    config_type: type[Conv1d.Config | Conv2d.Config | Conv3d.Config],
    given: str,
    inferred: str,
) -> None:
    config = config_type()
    setattr(config, given, 5)
    config.finalize()
    assert getattr(config, inferred) == 5


@pytest.mark.parametrize(
    ("config_type", "ndim"),
    [(Conv1d.Config, 1), (Conv2d.Config, 2), (Conv3d.Config, 3)],
)
def test_conv_reset_draws_the_bias_with_its_own_initializer(
    config_type: type[Conv1d.Config | Conv2d.Config | Conv3d.Config],
    ndim: int,
) -> None:
    config = config_type(2, 3)
    config.bias = True
    config.init_bias = torch.nn.init.ones_
    module = config.make()
    assert module.bias is not None
    assert torch.equal(module.bias, torch.ones(3))
    assert module.weight.ndim == ndim + 2


@pytest.mark.parametrize(
    ("config_type", "ndim"),
    [(Conv1d.Config, 1)],
)
def test_conv_reset_uses_torch_bias_draw_when_initializer_is_none(
    config_type: type[Conv1d.Config],
    ndim: int,
) -> None:
    """A missing custom bias initializer retains torch's fan-in uniform reset."""
    config = config_type(2, 3)
    config.bias = True
    config.init_bias = None
    module = config.make()
    module.reset_parameters()
    assert module.bias is not None
    fan_in = module.weight.shape[1] * math.prod(module.weight.shape[2:])
    bound = fan_in**-0.5

    assert module.weight.ndim == ndim + 2
    assert torch.all(module.bias >= -bound)
    assert torch.all(module.bias <= bound)
    assert torch.any(module.bias != 0)


def test_conv_cost_rejects_an_unknown_padding_string() -> None:
    with pytest.raises(TypeError, match=r"^Unsupported convolution padding: 'full'\.$"):
        conv_cost(
            channels_in=2,
            channels_out=3,
            kernel_size=3,
            ndim=1,
            groups=1,
            bias=False,
            input_grid=(4,),
            batch_size=2,
            padding="full",
        )


def test_conv_cost_rejects_a_grid_of_the_wrong_rank() -> None:
    with pytest.raises(ValueError, match=r"^input_grid has 2 axes; ndim is 1\.$"):
        conv_cost(
            channels_in=2,
            channels_out=3,
            kernel_size=3,
            ndim=1,
            groups=1,
            bias=False,
            input_grid=(4, 5),
            batch_size=2,
        )


@pytest.mark.parametrize(
    ("batch_size", "input_grid", "error", "message"),
    [
        (2.0, (4,), TypeError, "must be integers"),
        (2, (2,), ValueError, "must be positive"),
    ],
)
def test_conv_cost_rejects_row_counts_that_are_not_positive_integers(
    batch_size: int,
    input_grid: tuple[int, ...],
    error: type[Exception],
    message: str,
) -> None:
    """A float batch is not a row count; a kernel wider than its input leaves none."""
    with pytest.raises(error, match=message):
        conv_cost(
            channels_in=2,
            channels_out=3,
            kernel_size=3,
            ndim=1,
            groups=1,
            bias=False,
            input_grid=input_grid,
            batch_size=batch_size,
            padding="valid",
        )


def test_conv2d_cost_is_a_matmul_over_the_receptive_field() -> None:
    """Count every output position: ``channels_in * prod(kernel_size)`` -> ``channels_out``.

    ``padding="same"`` keeps every input position as an output position, so
    a ``4 x 6`` image is 24 tokens.
    """
    analytical = assert_cost_matches_torch(
        Conv2d.Config(2, 3, kernel_size=(3, 5), bias=True),
        build_input=lambda: torch.randn(5, 2, 4, 6, requires_grad=True),
        input_grid=(4, 6),
        batch_size=5,
        dtype=None,
    )
    weights = 3 * 2 * 15
    assert analytical["flops", "primal", "matmul"].sum() == 5 * 2 * 24 * weights
    assert analytical["flops", "adjoint", "matmul"].sum() == 5 * 4 * 24 * weights
    assert analytical.params == weights + 3
    assert analytical["bytes", "primal", "elementwise"].sum() == 4 * 3


def test_conv1d_cost_divides_the_fan_in_by_groups() -> None:
    """Grouped convolution counts only its connected input channels in both passes."""
    analytical = assert_cost_matches_torch(
        Conv1d.Config(4, channels_out=6, kernel_size=3, groups=2),
        build_input=lambda: torch.randn(3, 4, 7, requires_grad=True),
        input_grid=(7,),
        batch_size=3,
        dtype=None,
    )
    weights = 6 * (4 // 2) * 3
    assert analytical["flops", "primal", "matmul"].sum() == 3 * 2 * 7 * weights
    assert analytical.params == weights
    assert analytical["bytes", "primal", "matmul"].sum() == 4 * (
        3 * 7 * 4 + 3 * 7 * 6 + weights
    )


def test_conv3d_cost_cubes_a_scalar_kernel() -> None:
    analytical = assert_cost_matches_torch(
        Conv3d.Config(2, 3, kernel_size=3),
        build_input=lambda: torch.randn(6, 2, 3, 4, 5, requires_grad=True),
        input_grid=(3, 4, 5),
        batch_size=6,
        dtype=None,
    )
    assert analytical["flops", "primal", "matmul"].sum() == 6 * 2 * (3 * 4 * 5) * (
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
    from priml.lib.testing.runner import test_main

    test_main(__file__)
