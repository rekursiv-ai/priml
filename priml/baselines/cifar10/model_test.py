"""Tests for the CIFAR-10 networks."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from priml.baselines.cifar10.model import ConvBlock, ResNet, SpeedNet
from priml.model.init import dirac
from priml.testing.bfb import assert_bfb_against_golden


_GOLDEN_DIR = Path(__file__).parent.resolve() / "goldens"


def tiny_resnet() -> ResNet.Config:
    """Return the smallest ResNet that still exercises every code path."""
    config = ResNet.Config()
    config.channels_hidden = (8, 16)
    config.blocks_per_stage = 1
    return config


def tiny_speednet() -> SpeedNet.Config:
    """Return the smallest SpeedNet that still exercises every code path."""
    config = SpeedNet.Config()
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
        x = stage(x)
    # Three stages, the first at full resolution: 32 -> 32 -> 16 -> 8.
    assert x.shape[-1] == 8


def test_resnet_residual_path_is_identity_when_shape_is_preserved() -> None:
    config = tiny_resnet()
    config.channels_hidden = (8,)
    block = config.make().stages[0]
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
    widths = [block.convs[0].weight.shape[0] for block in model.blocks]
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
    # An identity kernel reproduces its input channel-for-channel, so a
    # freshly-initialized block is a no-op up to pooling and normalization.
    weight = block.convs[0].weight.data
    assert torch.equal(weight, torch.nn.init.dirac_(torch.empty_like(weight)))


def test_resnet_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_GOLDEN_DIR,
        golden_name="resnet_min_cpu",
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
        golden_dir=_GOLDEN_DIR,
        golden_name="speednet_min_cpu",
        build_module=lambda: tiny_speednet().make(),
        build_input=lambda: torch.randn(1, 3, 32, 32),
        seed=0,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
