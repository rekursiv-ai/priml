"""Tests for the CIFAR-10 networks."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

from configgle import InlineConfig
from torch import Tensor

import pytest
import torch

from priml.baselines.cifar10.model import (
    ConvBlock,
    ResidualBlock,
    ResNet,
    ScaledLinear,
    SpeedNet,
    _activation,
)
from priml.model.init import dirac
from priml.model.norm import BatchNorm2d
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


def tiny_resnet() -> ResNet.Config:
    """Return the smallest ResNet that still exercises every code path."""
    config = ResNet.Config(channels_in=3, channels_out=2)
    config.channels_hidden = (2, 4)
    config.blocks_per_stage = 1
    return config


def tiny_speednet() -> SpeedNet.Config:
    """Return the smallest SpeedNet that still exercises every code path."""
    config = SpeedNet.Config(channels_in=3, channels_out=2)
    config.image_size = (7, 8)
    config.channels_hidden = (4,)
    block = config.block = ConvBlock.Config()
    block.num_convs = 1
    return config


def test_activation_builds_makeable_config() -> None:
    activation = _activation(InlineConfig(torch.nn.PReLU))
    assert isinstance(activation, torch.nn.PReLU)


def test_activation_returns_plain_callable_by_identity() -> None:
    assert _activation(torch.relu) is torch.relu


def test_resnet_forward_shape() -> None:
    model = tiny_resnet().make()
    assert model(torch.randn(2, 3, 4, 5)).shape == (2, 2)


def test_resnet_downsamples_once_per_stage_after_the_first() -> None:
    config = tiny_resnet()
    config.channels_hidden = (2, 3, 4)
    model = config.make()
    x = model.stem(torch.randn(2, 3, 31, 32))
    for stage in model.stages:
        x = cast(Tensor, stage(x))
    # Three stages, the first at full resolution: 32 -> 32 -> 16 -> 8.
    assert x.shape[-1] == 8


def test_resnet_residual_path_is_identity_when_shape_is_preserved() -> None:
    config = tiny_resnet()
    config.channels_hidden = (2,)
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
    model.init_whiten(torch.randn(6, 3, 7, 8))
    assert model(torch.randn(2, 3, 7, 8)).shape == (2, 2)


def test_speednet_whitening_weights_are_frozen() -> None:
    model = tiny_speednet().make()
    assert not model.whiten.weight.requires_grad


def test_speednet_forwards_the_injected_pca_decomposition() -> None:
    model = tiny_speednet().make()
    decomposed: list[Tensor] = []

    def decompose(centered: Tensor) -> tuple[Tensor, Tensor]:
        decomposed.append(centered)
        width = centered.shape[1]
        return torch.ones(width), torch.eye(width)

    model.init_whiten(torch.randn(2, 3, 7, 8), decompose=decompose)

    assert [matrix.shape for matrix in decomposed] == [(2 * 6 * 7, 12)]
    scale = torch.rsqrt(torch.tensor(1.0 + 5e-4))
    # SpeedNet's whitening kernel is the production 2x2 image kernel.
    expected = scale * torch.eye(12).reshape(12, 3, 2, 2)
    assert torch.equal(model.whiten.weight[:12], expected)


def test_speednet_whitening_is_rank_doubled() -> None:
    config = tiny_speednet()
    config.whiten_kernel = 2
    model = config.make()
    model.init_whiten(torch.randn(8, 3, 25, 26))
    kernel = model.whiten.weight.data
    half = kernel.shape[0] // 2
    # The layer emits each eigenvector and its negation, so a following
    # activation can respond to projections of either sign.
    assert torch.equal(kernel[:half], -kernel[half:])


def test_speednet_block_list_must_match_the_stage_count() -> None:
    config = tiny_speednet()
    config.block = [ConvBlock.Config(), ConvBlock.Config()]
    with pytest.raises(ValueError, match="block list must hold 1 configs"):
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
    model.init_whiten(torch.randn(6, 3, 7, 8))
    assert model(torch.randn(2, 3, 7, 8)).shape == (2, 2)


def test_speednet_two_convs_omit_the_residual() -> None:
    config = tiny_speednet()
    block = config.block = ConvBlock.Config()
    block.num_convs = 2
    model = config.make()
    model.init_whiten(torch.randn(6, 3, 7, 8))
    assert model(torch.randn(2, 3, 7, 8)).shape == (2, 2)


def test_speednet_rejects_invalid_num_convs() -> None:
    config = tiny_speednet()
    block = config.block = ConvBlock.Config()
    block.num_convs = 4
    with pytest.raises(ValueError, match="num_convs must be 1, 2, or 3"):
        _ = config.make()


def test_speednet_dirac_init_passes_input_through_each_block() -> None:
    config = tiny_speednet()
    config.channels_hidden = (2, 2, 2)
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
        build_module=tiny_resnet().make,
        build_input=lambda: torch.randn(2, 3, 4, 5),
        seed=0,
    )


def test_speednet_bfb() -> None:
    # Whitening gives 6x7, the block pool gives 3x3, and the final 3x3 pool gives 1x1.
    # ``init_whiten`` is deliberately not called -- the harness overwrites every
    # parameter, the whitening kernel included, so the golden pins the forward
    # arithmetic rather than the PCA fit.
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="speednet",
        build_module=tiny_speednet().make,
        build_input=lambda: torch.randint(0, 256, (2, 3, 7, 8), dtype=torch.uint8),
        seed=0,
        run=lambda module, image: cast(SpeedNet, module)(image.float()),
    )


# Cost tests feed an input that REQUIRES grad, as ``conv_test.py`` does: priml
# costs every layer's full adjoint, whereas a real image carries no gradient
# and torch then skips the gradient into the first trainable layer's input.


def test_scaled_linear_cost_is_a_matmul_plus_one_scale_per_logit() -> None:
    analytical = assert_cost_matches_torch(
        ScaledLinear.Config(channels_in=6, channels_out=4),
        build_input=lambda: torch.randn(3, 6, requires_grad=True),
        seq_len=3,
        batch_size=1,
        dtype=None,
    )
    assert analytical.params == 6 * 4
    # ``1 / fan_in`` is a Python float, so the scale is one multiply per logit
    # each way and no parameter.
    assert analytical["flops", "primal", "elementwise"].sum() == 3 * 4
    assert analytical["flops", "adjoint", "elementwise"].sum() == 3 * 4


def test_residual_block_cost_prices_the_convolutions_at_the_strided_grid() -> None:
    """A stride-2 block runs its convolutions on a quarter of the input positions."""
    analytical = assert_cost_matches_torch(
        ResidualBlock.Config(
            channels_in=4,
            channels_out=6,
            stride=2,
            image_size=(6, 8),
        ),
        build_input=lambda: torch.randn(2, 4, 6, 8, requires_grad=True),
        seq_len=6 * 8,
        batch_size=2,
        dtype=None,
    )
    conv1, conv2, shortcut = 6 * 4 * 9, 6 * 6 * 9, 6 * 4
    assert analytical["flops", "primal", "matmul"].sum() == 2 * 2 * 3 * 4 * (
        conv1 + conv2 + shortcut
    )
    assert analytical.params == conv1 + conv2 + shortcut + 2 * 4 + 2 * 6
    # Elementwise: norm1 and one ReLU compare per input channel at the input
    # grid; norm2, a ReLU, and the residual add per output channel at the
    # output grid. The adjoint adds one accumulation per input channel where the
    # shortcut's and the branch's gradients meet.
    norm1 = BatchNorm2d.Config(4, elementwise_affine=True).cost(
        seq_len=96,
        batch_size=1,
        dtype=None,
    )
    norm2 = BatchNorm2d.Config(6, elementwise_affine=True).cost(
        seq_len=24,
        batch_size=1,
        dtype=None,
    )
    assert analytical["flops", "primal", "elementwise"].sum() == (
        norm1["flops", "primal", "elementwise"].sum()
        + 4 * 96
        + norm2["flops", "primal", "elementwise"].sum()
        + 6 * 24
        + 6 * 24
    )
    assert analytical["flops", "adjoint", "elementwise"].sum() == (
        norm1["flops", "adjoint", "elementwise"].sum()
        + 4 * 96
        + 4 * 96
        + norm2["flops", "adjoint", "elementwise"].sum()
        + 6 * 24
    )


def test_residual_block_cost_omits_the_shortcut_when_shape_is_preserved() -> None:
    # ResidualBlock cost resolves a missing image_size through _block_grids, which requires a square grid.
    analytical = assert_cost_matches_torch(
        ResidualBlock.Config(channels_in=5, channels_out=5),
        build_input=lambda: torch.randn(2, 5, 4, 4, requires_grad=True),
        seq_len=4 * 4,
        batch_size=2,
        dtype=None,
    )
    assert analytical.params == 2 * (5 * 5 * 9) + 2 * (2 * 5)


def test_conv_block_cost_pools_after_the_first_convolution() -> None:
    """The first convolution runs at the input grid, the other two at a quarter of it."""
    block = ConvBlock.Config(channels_in=4, channels_out=6)
    block.num_convs = 3
    block.image_size = (6, 8)
    analytical = assert_cost_matches_torch(
        block,
        build_input=lambda: torch.randn(2, 4, 6, 8, requires_grad=True),
        seq_len=6 * 8,
        batch_size=2,
        dtype=None,
    )
    first, later = 6 * 4 * 9, 6 * 6 * 9
    assert (
        analytical["flops", "primal", "matmul"].sum()
        == 2 * first * 96 + 2 * 2 * later * 24
    )
    # ``affine=False`` norms own nothing.
    assert analytical.params == first + 2 * later
    # The 2x2 max pool is three compares per pooled channel forward and one
    # gradient element routed back to the argmax; the norms add their own sums.
    norm = BatchNorm2d.Config(6).cost(seq_len=12, batch_size=2, dtype=None)
    assert analytical["flops", "primal", "reduction"].sum() == (
        6 * 3 * 24 + 3 * norm["flops", "primal", "reduction"].sum()
    )
    assert analytical["flops", "adjoint", "selection"].sum() == 6 * 24


def test_resnet_cost_counts_each_stage_for_the_complete_batch() -> None:
    """Every matmul the forward issues is in the cost, and every parameter."""
    config = tiny_resnet()
    config.image_size = (6, 8)
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 6, 8, requires_grad=True),
        batch_size=2,
        dtype=None,
    )
    stem, stage0 = 2 * 3 * 9, 2 * (2 * 2 * 9)
    stage1 = 4 * 2 * 9 + 4 * 4 * 9 + 4 * 2
    head = 4 * 2
    assert analytical["flops", "primal", "matmul"].sum() == 2 * (
        stem * 96 + stage0 * 96 + stage1 * 24 + head * 2
    )
    # Five affine norms: three of width 2, two of width 4; the head owns a bias.
    assert analytical.params == (
        stem + stage0 + stage1 + head + 3 * (2 * 2) + 2 * (2 * 4) + 2
    )
    # Global average pooling sums 4x4 positions per channel once per image; the
    # four norms sum at their own grids.
    norm2 = BatchNorm2d.Config(2, elementwise_affine=True).cost(
        seq_len=48,
        batch_size=2,
        dtype=None,
    )
    norm4 = BatchNorm2d.Config(4, elementwise_affine=True).cost(
        seq_len=12,
        batch_size=2,
        dtype=None,
    )
    assert analytical["flops", "primal", "reduction"].sum() == (
        3 * norm2["flops", "primal", "reduction"].sum()
        + 2 * norm4["flops", "primal", "reduction"].sum()
        + 4 * 2 * (12 - 1)
    )


def test_speednet_cost_prices_the_frozen_whitening_and_every_pool() -> None:
    """Every matmul the forward issues is in the cost, and every parameter.

    The whitening weight is frozen but owned: ``parameters()`` lists it, so
    it counts, while its adjoint forms only the input gradient.
    """
    config = tiny_speednet()
    config.image_size = (7, 8)
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 7, 8, requires_grad=True),
        batch_size=2,
        dtype=None,
    )
    whiten = 24 * 3 * 4
    block = 4 * 24 * 9
    head = 2 * 4
    assert analytical.params == whiten + block + head
    # 7x8 -> 6x7 (whiten) -> 3x3 (block pool) -> 1x1 (final pool).
    assert analytical["flops", "primal", "matmul"].sum() == 2 * 2 * (
        whiten * 6 * 7 + block * 6 * 7 + head
    )
    assert analytical["flops", "adjoint", "matmul"].sum() == 2 * 2 * (
        whiten * 6 * 7 + 2 * (block * 6 * 7 + head)
    )
    # One gradient element routed back per pooled channel: the block pool and
    # the final 3x3 pool.
    assert analytical["flops", "adjoint", "selection"].sum() == 2 * (4 * 3 * 3 + 4 * 1)


@pytest.mark.parametrize("speednet", [False, True])
def test_image_cost_scales_bytes_not_flops_with_itemsize(speednet: bool) -> None:
    config = (tiny_speednet() if speednet else tiny_resnet()).finalize()
    wide = config.cost(batch_size=2, dtype=torch.float64)
    narrow = config.cost(batch_size=2, dtype=torch.bfloat16)
    assert wide["flops", "primal"].sum() == narrow["flops", "primal"].sum()
    assert wide["flops", "adjoint"].sum() == narrow["flops", "adjoint"].sum()
    # Saved argmax indices stay int64 at either width; payload cells scale.
    assert (
        wide["bytes", torch.float64].sum() == narrow["bytes", torch.bfloat16].sum() * 4
    )
    assert wide["bytes", torch.int64] == narrow["bytes", torch.int64]


def test_scaled_linear_traffic_counts_scale_input_and_output() -> None:
    config = ScaledLinear.Config()
    config.channels_in = 6
    config.channels_out = 4
    costed = config.cost(seq_len=3, batch_size=1, dtype=torch.bfloat16)
    assert costed["bytes", "primal", "elementwise"].sum() == 3 * 2 * (4 + 4)
    assert costed["bytes", "adjoint", "elementwise"].sum() == 3 * 2 * (4 + 4)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
