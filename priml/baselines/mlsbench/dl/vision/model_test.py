"""Tests for the residual network."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from configgle import InlineConfig, Makeable
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.mlsbench.dl.vision.model import (
    Activation,
    BasicBlock,
    ResNet,
    activation_layer,
    init_layers,
    kaiming_normal_fan_in,
    kaiming_normal_fan_out,
    sized_blocks,
)
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from collections.abc import Sequence


_CWD: Final = Path(__file__).resolve().parent

BATCH: Final = 2
CLASSES: Final = 7
HEIGHT: Final = 8
WIDTH: Final = 10


def tiny() -> ResNet.Config:
    """Return the smallest network that still has three stages."""
    config = ResNet.Config()
    config.channels_in = 3
    config.channels_out = CLASSES
    config.channels_hidden = (4, 5, 6)
    config.blocks_per_stage = (1, 1, 1)
    return config


def blocks(model: ResNet) -> list[list[BasicBlock]]:
    """Return the blocks of each stage, typed."""
    return [[cast(BasicBlock, block) for block in stage] for stage in model.stages]


def registered(prefix: str, *, projected: bool) -> list[str]:
    """Return the parameter names of one block, in the order the reference has them."""
    names = [
        f"{prefix}.conv1.weight",
        f"{prefix}.norm1.weight",
        f"{prefix}.norm1.bias",
        f"{prefix}.conv2.weight",
        f"{prefix}.norm2.weight",
        f"{prefix}.norm2.bias",
    ]
    if projected:
        names += [
            f"{prefix}.shortcut.0.weight",
            f"{prefix}.shortcut.1.weight",
            f"{prefix}.shortcut.1.bias",
        ]
    return names


def stage_shapes(model: ResNet, media: Tensor) -> list[tuple[int, ...]]:
    """Return the shape each stage hands to the next."""
    x = model.act(cast(Tensor, model.norm(model.stem(media))))
    shapes: list[tuple[int, ...]] = []
    for stage in model.stages:
        x = stage(x)
        shapes.append(tuple(x.shape))
    return shapes


def test_forward_returns_one_score_per_class() -> None:
    model = tiny().make()
    assert model(torch.randn(BATCH, 3, HEIGHT, WIDTH)).shape == (BATCH, CLASSES)


def test_each_stage_after_the_first_halves_the_resolution_and_sets_its_width() -> None:
    model = tiny().make()
    assert stage_shapes(model, torch.randn(BATCH, 3, HEIGHT, WIDTH)) == [
        (BATCH, 4, 8, 10),
        (BATCH, 5, 4, 5),
        (BATCH, 6, 2, 3),
    ]


def test_every_stage_holds_its_own_number_of_blocks_and_strides_only_its_first() -> (
    None
):
    config = tiny()
    config.blocks_per_stage = (2, 1, 3)
    model = config.make()
    assert [len(stage) for stage in blocks(model)] == [2, 1, 3]
    strides = [[block.conv1.stride for block in stage] for stage in blocks(model)]
    assert strides == [[(1, 1)] * 2, [(2, 2)], [(2, 2), (1, 1), (1, 1)]]


def test_projection_shortcuts_appear_only_where_the_shape_changes() -> None:
    model = tiny().make()
    first = [stage[0] for stage in blocks(model)]
    assert isinstance(first[0].shortcut, nn.Identity)
    for block in first[1:]:
        assert isinstance(block.shortcut, nn.Sequential)
        conv = cast(nn.Conv2d, block.shortcut[0])
        assert (conv.kernel_size, conv.stride) == ((1, 1), (2, 2))


def test_a_block_adds_its_shortcut_before_the_last_activation() -> None:
    block = BasicBlock.Config(channels_in=4, channels_out=4).make().eval()
    with torch.no_grad():
        block.conv1.weight.zero_()
        block.conv2.weight.zero_()
    x = torch.randn(BATCH, 4, 3, 5)
    assert (x < 0).any()
    assert torch.equal(block(x), torch.relu(x))


def test_layers_are_registered_in_the_reference_order() -> None:
    expected = ["stem.weight", "norm.weight", "norm.bias"]
    expected += registered("stages.0.0", projected=False)
    expected += registered("stages.1.0", projected=True)
    expected += registered("stages.2.0", projected=True)
    expected += ["head.weight", "head.bias"]
    assert [name for name, _ in tiny().make().named_parameters()] == expected


def test_default_initialization_is_he_normal_with_a_zero_head_bias() -> None:
    model = tiny().make()
    assert torch.eq(cast(Tensor, model.head.bias), 0).all()
    norm = cast(nn.BatchNorm2d, model.norm)
    assert torch.eq(norm.weight, 1).all()
    assert torch.eq(norm.bias, 0).all()
    assert model.stem.weight.std() > 0


def test_initializer_slots_reach_the_layers_they_name() -> None:
    config = tiny()
    config.init_conv = nn.init.ones_
    config.init_head = nn.init.zeros_
    config.init_bias = nn.init.ones_
    model = config.make()
    assert all(
        torch.eq(m.weight, 1).all() for m in model.modules() if isinstance(m, nn.Conv2d)
    )
    assert torch.eq(model.head.weight, 0).all()
    assert torch.eq(cast(Tensor, model.head.bias), 1).all()


def test_a_function_activation_is_wrapped_as_a_layer() -> None:
    layer = activation_layer(torch.sigmoid)
    assert isinstance(layer, Activation)
    x = torch.tensor([-1.0, 0.0, 2.0])
    assert torch.equal(layer(x), torch.sigmoid(x))


def test_a_module_activation_is_its_own_layer() -> None:
    layer = activation_layer(InlineConfig(nn.PReLU))
    assert isinstance(layer, nn.PReLU)
    assert activation_layer(InlineConfig(nn.PReLU)) is not layer


def test_init_layers_gives_each_layer_kind_its_own_initializer() -> None:
    conv, head, bare = nn.Conv2d(2, 3, 1), nn.Linear(3, 2), nn.Linear(2, 2, bias=False)
    conv_bias = cast(Tensor, conv.bias).clone()
    init_layers(
        nn.Sequential(conv, head, bare),
        init_conv=nn.init.ones_,
        init_head=nn.init.zeros_,
        init_bias=nn.init.ones_,
    )
    assert torch.eq(conv.weight, 1).all()
    assert torch.equal(cast(Tensor, conv.bias), conv_bias)
    assert torch.eq(head.weight, 0).all()
    assert torch.eq(cast(Tensor, head.bias), 1).all()
    assert torch.eq(bare.weight, 0).all()
    assert bare.bias is None


def test_fan_out_and_fan_in_scale_a_convolution_differently() -> None:
    # A 3x3 kernel, as every convolution of ``BasicBlock`` has.
    weight = torch.empty(64, 32, 3, 3)
    torch.manual_seed(0)
    kaiming_normal_fan_out(weight)
    assert weight.std() == pytest.approx((2 / (64 * 9)) ** 0.5, rel=0.05)
    kaiming_normal_fan_in(weight)
    assert weight.std() == pytest.approx((2 / (32 * 9)) ** 0.5, rel=0.05)


def test_the_activation_slot_reaches_the_stem_and_every_block() -> None:
    config = tiny()
    config.activation = torch.sigmoid
    model = config.make()
    assert model.act is torch.sigmoid
    assert all(b.act is torch.sigmoid for stage in blocks(model) for b in stage)


def test_a_stateful_activation_is_built_once_per_use() -> None:
    config = tiny()
    config.activation = InlineConfig(nn.PReLU)
    model = config.make()
    acts = [model.act, *(b.act for stage in blocks(model) for b in stage)]
    assert all(isinstance(act, nn.PReLU) for act in acts)
    assert len({id(act) for act in acts}) == len(acts)


def test_the_norm_slot_reaches_the_stem_every_block_and_every_projection() -> None:
    config = tiny()
    config.norm = nn.InstanceNorm2d
    model = config.make()
    norms = [model.norm] + [
        layer
        for stage in blocks(model)
        for b in stage
        for layer in (b.norm1, b.norm2, *b.shortcut.children())
        if not isinstance(layer, nn.Conv2d)
    ]
    assert len(norms) == 1 + 2 * 3 + 2
    assert all(type(norm) is nn.InstanceNorm2d for norm in norms)


def test_the_pool_slot_feeds_the_head_and_registers_before_it() -> None:
    config = tiny()
    config.pool = InlineConfig(nn.Sequential, nn.AdaptiveMaxPool2d(1), nn.Flatten())
    model = config.make()
    media = torch.randn(BATCH, 3, HEIGHT, WIDTH)
    features = model.stages(model.act(cast(Tensor, model.norm(model.stem(media)))))
    assert torch.equal(model(media), model.head(features.amax(dim=(-2, -1))))
    assert list(dict(model.named_children())) == [
        "stem",
        "norm",
        "stages",
        "pool",
        "head",
    ]


def test_the_init_hook_runs_after_the_per_tensor_initializers() -> None:
    seen: list[Tensor] = []
    config = tiny()
    config.init_conv = nn.init.ones_
    config.init = lambda model: seen.append(cast(ResNet, model).stem.weight.clone())
    _ = config.make()
    (stem,) = seen
    assert torch.eq(stem, 1).all()


class Numbered(BasicBlock.Config):
    """A basic block template that numbers its sized copies in network order."""

    def place(self, blocks: Sequence[Makeable[nn.Module]]) -> None:
        """Give each copy its zero-based position as its stride's companion."""
        for index, block in enumerate(blocks):
            assert isinstance(block, Numbered)
            block.position = index

    position: int = -1


def test_a_positioned_template_sees_every_sized_copy_in_order() -> None:
    config = tiny()
    config.blocks_per_stage = (2, 1, 1)
    config.block = Numbered()
    stages = sized_blocks(config)
    copies = [block for stage in stages for block, _ in stage]
    assert [cast(Numbered, b).position for b in copies] == [0, 1, 2, 3]
    assert [cast(Numbered, b).channels_out for b in copies] == [4, 4, 5, 6]
    assert config.block.position == -1


def test_building_leaves_the_block_template_untouched() -> None:
    config = tiny()
    _ = config.make()
    assert isinstance(config.block, BasicBlock.Config)
    assert (config.block.channels_in, config.block.channels_out) == (-1, -1)


def test_blocks_report_their_widths_from_the_stage() -> None:
    model = tiny().make()
    by_stage = blocks(model)
    assert by_stage[1][0].conv1.weight.shape == (5, 4, 3, 3)
    assert by_stage[2][0].conv2.weight.shape == (6, 6, 3, 3)


def test_a_block_infers_the_width_it_is_not_given() -> None:
    config = BasicBlock.Config(channels_in=4)
    assert config.copy_tree().finalize().channels_out == 4
    config = BasicBlock.Config(channels_out=5)
    assert config.copy_tree().finalize().channels_in == 5


def test_every_stage_needs_a_block() -> None:
    config = tiny()
    config.blocks_per_stage = (1, 0, 1)
    with pytest.raises(ValueError, match="Every stage needs a block"):
        config.make()


# Cost tests feed an input that REQUIRES grad, as ``conv_test.py`` does: priml costs
# every layer's full adjoint, whereas a real image carries no gradient and torch then
# skips the gradient into the first trainable layer's input. A block is costed on a
# square grid, so those inputs are square: BasicBlock.Config.cost reads one side from
# seq_len.


def test_a_projected_block_runs_its_convolutions_on_the_strided_grid() -> None:
    analytical = assert_cost_matches_torch(
        # ``BasicBlock.cost`` takes a square grid only.
        BasicBlock.Config(channels_in=4, channels_out=6, stride=2),
        build_input=lambda: torch.randn(BATCH, 4, 6, 6, requires_grad=True),
        seq_len=6 * 6,
        batch_size=BATCH,
        dtype=None,
    )
    conv1, conv2, shortcut = 6 * 4 * 9, 6 * 6 * 9, 6 * 4
    assert analytical["flops", "primal", "matmul"].sum() == 2 * BATCH * 3 * 3 * (
        conv1 + conv2 + shortcut
    )
    assert analytical.params == conv1 + conv2 + shortcut + 3 * 2 * 6


def test_an_identity_block_owns_no_shortcut() -> None:
    analytical = assert_cost_matches_torch(
        # ``BasicBlock.cost`` takes a square grid only.
        BasicBlock.Config(channels_in=5, channels_out=5),
        build_input=lambda: torch.randn(BATCH, 5, 4, 4, requires_grad=True),
        seq_len=4 * 4,
        batch_size=BATCH,
        dtype=None,
    )
    assert analytical.params == 2 * (5 * 5 * 9) + 2 * (2 * 5)


def test_a_block_cannot_be_costed_on_a_grid_that_is_not_square() -> None:
    with pytest.raises(ValueError, match="square"):
        BasicBlock.Config(channels_in=4, channels_out=4).cost(
            seq_len=6,
            batch_size=BATCH,
            dtype=None,
        )


def test_the_network_cost_counts_every_matmul_the_forward_issues() -> None:
    config = tiny()
    # ``ResNet.cost`` reads one ``image_size``: the grid is square.
    config.image_size = 8
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(BATCH, 3, 8, 8, requires_grad=True),
        batch_size=BATCH,
        dtype=None,
    )
    stem, stage0 = 3 * 4 * 9, 2 * (4 * 4 * 9)
    stage1, stage2 = 5 * 4 * 9 + 5 * 5 * 9 + 5 * 4, 6 * 5 * 9 + 6 * 6 * 9 + 6 * 5
    head = 6 * CLASSES
    assert analytical["flops", "primal", "matmul"].sum() == 2 * BATCH * (
        (stem + stage0) * 8 * 8 + stage1 * 4 * 4 + stage2 * 2 * 2 + head
    )
    norms = 2 * (4 + 4 + 4 + 5 + 5 + 5 + 6 + 6 + 6)
    assert analytical.params == stem + stage0 + stage1 + stage2 + head + CLASSES + norms


def test_resnet_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="resnet",
        build_module=tiny().make,
        build_input=lambda: torch.randn(BATCH, 3, HEIGHT, WIDTH),
        seed=0,
    )


def test_features_are_what_the_head_reads() -> None:
    model = tiny().make()
    media = torch.randn(BATCH, 3, HEIGHT, WIDTH)
    features = model.features(media)
    assert features.shape == (BATCH, 6)
    assert torch.equal(model(media), model.head(features))


def test_an_empty_network_is_refused_by_name() -> None:
    config = tiny()
    config.blocks_per_stage = ()
    config.channels_hidden = ()
    with pytest.raises(ValueError, match="Every stage needs a block"):
        config.make()


def test_stage_lists_of_unequal_length_are_refused_by_name() -> None:
    config = tiny()
    config.blocks_per_stage = (1, 1)
    with pytest.raises(ValueError, match="2 stage depths for 3 stage widths"):
        config.make()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
