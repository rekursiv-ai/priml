"""Tests for the CIFAR-10 networks."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import pytest
import torch

from priml.baselines.cifar10.model import (
    ConvBlock,
    ResidualBlock,
    ResNet,
    ScaledLinear,
    SpeedNet,
    _max_pool_cost,
)
from priml.model.init import dirac
from priml.model.norm import BatchNorm2d
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


def tiny_resnet() -> ResNet.Config:
    """Return the smallest ResNet that still exercises every code path."""
    config = ResNet.Config(channels_in=3, channels_out=10)
    config.channels_hidden = (8, 16)
    config.blocks_per_stage = 1
    return config


def tiny_speednet() -> SpeedNet.Config:
    """Return the smallest SpeedNet that still exercises every code path."""
    config = SpeedNet.Config(channels_in=3, channels_out=10)
    config.channels_hidden = (8, 16, 24)
    block = config.block = ConvBlock.Config()
    block.num_convs = 1
    return config


def test_resnet_forward_shape() -> None:
    model = tiny_resnet().make()
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_resnet_downsamples_once_per_stage_after_the_first() -> None:
    config = tiny_resnet()
    config.channels_hidden = (8, 16, 32)
    model = config.make()
    x = model.stem(torch.randn(1, 3, 32, 32))
    for stage in model.stages:
        x = cast(torch.Tensor, stage(x))
    # Three stages, the first at full resolution: 32 -> 32 -> 16 -> 8.
    assert x.shape[-1] == 8


def test_resnet_residual_path_is_identity_when_shape_is_preserved() -> None:
    config = tiny_resnet()
    config.channels_hidden = (8,)
    block = config.make().stages[0]
    assert isinstance(block, ResidualBlock)
    assert isinstance(block.shortcut, torch.nn.Identity)


def test_resnet_rejects_empty_channels_hidden() -> None:
    config = tiny_resnet()
    config.channels_hidden = ()
    with pytest.raises(ValueError, match="at least one stage"):
        _ = config.make()


def test_resnet_rejects_zero_blocks() -> None:
    config = tiny_resnet()
    config.blocks_per_stage = 0
    with pytest.raises(ValueError, match="blocks_per_stage must be positive"):
        _ = config.make()


def test_speednet_forward_shape() -> None:
    model = tiny_speednet().make()
    model.init_whiten(torch.randn(8, 3, 32, 32))
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_speednet_whitening_weights_are_frozen() -> None:
    model = tiny_speednet().make()
    assert not model.whiten.weight.requires_grad


def test_speednet_whitening_is_rank_doubled() -> None:
    config = tiny_speednet()
    config.whiten_kernel = 2
    model = config.make()
    model.init_whiten(torch.randn(8, 3, 32, 32))
    kernel = model.whiten.weight.data
    half = kernel.shape[0] // 2
    # The layer emits each eigenvector and its negation, so a following
    # activation can respond to projections of either sign.
    assert torch.equal(kernel[:half], -kernel[half:])


def test_speednet_block_list_must_match_the_stage_count() -> None:
    config = tiny_speednet()
    config.block = [ConvBlock.Config(), ConvBlock.Config()]
    with pytest.raises(ValueError, match="block list must hold 3 configs"):
        _ = config.make()


def test_speednet_width_follows_channels_hidden() -> None:
    """The template is copied per stage, so each block gets its own width."""
    config = tiny_speednet()
    model = config.make()
    widths: list[int] = []
    for block in model.blocks:
        assert isinstance(block, ConvBlock)
        widths.append(block.convs[0].weight.shape[0])
    assert widths == list(config.channels_hidden)


def test_speednet_three_convs_add_a_residual() -> None:
    config = tiny_speednet()
    block = config.block = ConvBlock.Config()
    block.num_convs = 3
    model = config.make()
    model.init_whiten(torch.randn(8, 3, 32, 32))
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_speednet_two_convs_omit_the_residual() -> None:
    config = tiny_speednet()
    block = config.block = ConvBlock.Config()
    block.num_convs = 2
    model = config.make()
    model.init_whiten(torch.randn(8, 3, 32, 32))
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_speednet_rejects_invalid_num_convs() -> None:
    config = tiny_speednet()
    block = config.block = ConvBlock.Config()
    block.num_convs = 4
    with pytest.raises(ValueError, match="num_convs must be 1, 2, or 3"):
        _ = config.make()


def test_speednet_dirac_init_passes_input_through_each_block() -> None:
    config = tiny_speednet()
    config.channels_hidden = (8, 8, 8)
    config.init_conv = dirac
    model = config.make()
    block = model.blocks[1]
    assert isinstance(block, ConvBlock)
    # An identity kernel reproduces its input channel-for-channel, so a
    # freshly-initialized block is a no-op up to pooling and normalization.
    weight = block.convs[0].weight.data
    assert torch.equal(weight, torch.nn.init.dirac_(torch.empty_like(weight)))


def test_resnet_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="resnet",
        build_module=lambda: tiny_resnet().make(),
        build_input=lambda: torch.randn(2, 3, 8, 8),
        seed=0,
    )


def test_speednet_bfb() -> None:
    # 32x32, not the 8x8 the ResNet golden uses: three pooling blocks and the
    # final MaxPool2d(3) reduce 32 -> 31 -> 15 -> 7 -> 3 -> 1, and anything
    # smaller pools away to nothing. ``init_whiten`` is deliberately not called
    # -- the harness overwrites every parameter, the whitening kernel included,
    # so the golden pins the forward arithmetic rather than the PCA fit.
    #
    # ONE image, because that 32x32 floor makes the input the largest thing in
    # the file. A second row would double it while re-checking arithmetic the
    # first row already covers -- there is no cross-batch interaction here to
    # catch, since BatchNorm runs with ``affine=False`` and the harness never
    # reaches training mode.
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="speednet",
        build_module=lambda: tiny_speednet().make(),
        build_input=lambda: torch.randn(1, 3, 32, 32),
        seed=0,
    )


# Cost tests feed an input that REQUIRES grad, as ``conv_test.py`` does: priml
# prices every layer's full adjoint, whereas a real image carries no gradient
# and torch then skips the gradient into the first trainable layer's input.
# Grids are powers of two so every ``channels / rows`` term inside a
# BatchNorm estimate is dyadic and the equalities below stay exact.


def test_scaled_linear_cost_is_a_matmul_plus_one_scale_per_logit() -> None:
    analytical = assert_cost_matches_torch(
        ScaledLinear.Config(channels_in=6, channels_out=4),
        build_input=lambda: torch.randn(3, 6, requires_grad=True),
        num_tokens=3,
    )
    assert analytical.params == 6 * 4
    # ``1 / fan_in`` is a Python float, so the scale is one multiply per logit
    # each way and no parameter.
    assert analytical.primal.flops.elementwise == 4
    assert analytical.adjoint.flops.elementwise == 4


def test_residual_block_cost_prices_the_convolutions_at_the_strided_grid() -> None:
    """A stride-2 block runs its convolutions on a quarter of the input positions."""
    analytical = assert_cost_matches_torch(
        ResidualBlock.Config(channels_in=4, channels_out=6, stride=2),
        build_input=lambda: torch.randn(2, 4, 8, 8, requires_grad=True),
        num_tokens=2 * 8 * 8,
        bus={"image_size": (8, 8), "batch_size": 2},
    )
    conv1, conv2, shortcut = 6 * 4 * 9, 6 * 6 * 9, 6 * 4
    assert analytical.primal.flops.matmul == 2 * (conv1 + conv2 + shortcut) / 4
    assert analytical.params == conv1 + conv2 + shortcut + 2 * 4 + 2 * 6
    # Elementwise: norm1 and one ReLU compare per input channel at the input
    # grid; norm2, a ReLU, and the residual add per output channel at the
    # output grid. The adjoint adds one accumulation per input channel where the
    # shortcut's and the branch's gradients meet.
    norm1 = BatchNorm2d.Config(4, elementwise_affine=True).cost(rows=128)
    norm2 = BatchNorm2d.Config(6, elementwise_affine=True).cost(rows=32)
    assert analytical.primal.flops.elementwise == (
        norm1.primal.flops.elementwise
        + 4
        + (norm2.primal.flops.elementwise + 6 + 6) / 4
    )
    assert analytical.adjoint.flops.elementwise == (
        norm1.adjoint.flops.elementwise
        + 4
        + 4
        + (norm2.adjoint.flops.elementwise + 6) / 4
    )


def test_residual_block_cost_omits_the_shortcut_when_shape_is_preserved() -> None:
    analytical = assert_cost_matches_torch(
        ResidualBlock.Config(channels_in=4, channels_out=4),
        build_input=lambda: torch.randn(2, 4, 4, 4, requires_grad=True),
        num_tokens=2 * 4 * 4,
        bus={"image_size": (4, 4), "batch_size": 2},
    )
    assert analytical.params == 2 * (4 * 4 * 9) + 2 * (2 * 4)


def test_conv_block_cost_pools_after_the_first_convolution() -> None:
    """The first convolution runs at the input grid, the other two at a quarter of it."""
    block = ConvBlock.Config(channels_in=4, channels_out=6)
    block.num_convs = 3
    analytical = assert_cost_matches_torch(
        block,
        build_input=lambda: torch.randn(2, 4, 8, 8, requires_grad=True),
        num_tokens=2 * 8 * 8,
        bus={"image_size": (8, 8), "batch_size": 2},
    )
    first, later = 6 * 4 * 9, 6 * 6 * 9
    assert analytical.primal.flops.matmul == 2 * first + 2 * 2 * later / 4
    # ``affine=False`` norms own nothing.
    assert analytical.params == first + 2 * later
    # The 2x2 max pool is three compares per pooled channel forward and one
    # gradient element routed back to the argmax; the norms add their own sums.
    norm = BatchNorm2d.Config(6).cost(rows=32)
    assert analytical.primal.flops.reduction == (
        (6 * 3 + 3 * norm.primal.flops.reduction) / 4
    )
    assert analytical.adjoint.flops.selection == 6 / 4


def test_resnet_cost_amortizes_each_stage_over_the_input_positions() -> None:
    """Every matmul the forward issues is in the cost, and every parameter."""
    analytical = assert_cost_matches_torch(
        tiny_resnet(),
        build_input=lambda: torch.randn(2, 3, 8, 8, requires_grad=True),
        num_tokens=2 * 8 * 8,
        bus={"image_size": (8, 8), "batch_size": 2},
    )
    stem, stage0 = 8 * 3 * 9, 2 * (8 * 8 * 9)
    stage1 = 16 * 8 * 9 + 16 * 16 * 9 + 16 * 8
    head = 16 * 10
    assert analytical.primal.flops.matmul == 2 * (
        stem + stage0 + stage1 / 4 + head / 64
    )
    # Five affine norms: three of width 8, two of width 16; the head owns a bias.
    assert analytical.params == (
        stem + stage0 + stage1 + head + 3 * (2 * 8) + 2 * (2 * 16) + 10
    )
    # Global average pooling sums 4x4 positions per channel once per image; the
    # four norms sum at their own grids.
    norm8 = BatchNorm2d.Config(8, elementwise_affine=True).cost(rows=128)
    norm16 = BatchNorm2d.Config(16, elementwise_affine=True).cost(rows=32)
    assert analytical.primal.flops.reduction == (
        3 * norm8.primal.flops.reduction
        + 2 * norm16.primal.flops.reduction / 4
        + 16 * (16 - 1) / 64
    )


def test_speednet_cost_prices_the_frozen_whitening_and_every_pool() -> None:
    """Every matmul the forward issues is in the cost, and every parameter.

    The whitening weight is frozen but owned: ``parameters()`` lists it, so
    it counts, while its adjoint forms only the input gradient.
    """
    analytical = assert_cost_matches_torch(
        tiny_speednet(),
        build_input=lambda: torch.randn(1, 3, 32, 32, requires_grad=True),
        num_tokens=32 * 32,
        bus={"image_size": (32, 32)},
    )
    whiten = 24 * 3 * 4
    blocks = (8 * 24 * 9, 16 * 8 * 9, 24 * 16 * 9)
    head = 10 * 24
    assert analytical.params == whiten + sum(blocks) + head
    # 32 -> 31 (whiten) -> 15 -> 7 -> 3 (block pools) -> 1 (final pool).
    assert (
        analytical.primal.flops.matmul
        == 2
        * (whiten * 961 + blocks[0] * 961 + blocks[1] * 225 + blocks[2] * 49 + head)
        / 1024
    )
    assert (
        analytical.adjoint.flops.matmul
        == 2
        * (
            whiten * 961
            + 2 * (blocks[0] * 961 + blocks[1] * 225 + blocks[2] * 49 + head)
        )
        / 1024
    )
    # One gradient element routed back per pooled channel: three block pools
    # and the final 3x3 pool.
    assert (
        analytical.adjoint.flops.selection
        == (8 * 225 + 16 * 49 + 24 * 9 + 24 * 1) / 1024
    )


@pytest.mark.parametrize("speednet", [False, True])
def test_image_cost_scales_bytes_not_flops_with_itemsize(speednet: bool) -> None:
    config = (tiny_speednet() if speednet else tiny_resnet()).finalize()
    wide = config.cost(image_size=(32, 32), batch_size=2, itemsize=8)
    narrow = config.cost(image_size=(32, 32), batch_size=2, itemsize=2)
    assert wide.primal.flops == narrow.primal.flops
    assert wide.adjoint.flops == narrow.adjoint.flops
    assert wide.training.bytes == narrow.training.bytes * 4


@pytest.mark.parametrize("itemsize", [2, 4, 8])
@pytest.mark.parametrize("kernel_size", [2, 3])
def test_max_pool_traffic_includes_argmax_and_dense_gradient(
    itemsize: int,
    kernel_size: int,
) -> None:
    priced = _max_pool_cost(3, kernel_size=kernel_size, itemsize=itemsize)
    elements = kernel_size**2
    assert priced.primal.bytes.reduction == itemsize * 3 * (elements + 2)
    assert priced.adjoint.bytes.selection == itemsize * 3 * (elements + 2)
    assert priced.primal.flops.reduction == 3 * (elements - 1)
    assert priced.adjoint.flops.selection == 3


def test_scaled_linear_traffic_counts_scale_input_and_output() -> None:
    config = ScaledLinear.Config()
    config.channels_in = 6
    config.channels_out = 4
    priced = config.cost(rows=3, itemsize=2)
    assert priced.primal.bytes.elementwise == 2 * (4 + 4)
    assert priced.adjoint.bytes.elementwise == 2 * (4 + 4)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
