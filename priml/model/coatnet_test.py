"""Check CoAtNet on small synthetic images."""

from __future__ import annotations

from typing import cast, override

import math

from configgle import Fig
from torch import nn
from torch.nn import functional

import pytest
import torch

from priml.cost import cost
from priml.model.attention.kernel import attention_kernel_cost
from priml.model.coatnet import (
    CoAtAttention,
    CoAtConvMLP,
    CoAtDownNorm,
    CoAtLinearMLP,
    CoAtMBConv,
    CoAtNet,
    CoAtNorm,
    CoAtPool,
    CoAtRelativeBias,
    CoAtScale,
    CoAtStage,
    CoAtStem,
    CoAtTransformer,
    silu_inplace,
)
from priml.model.conv import Conv2d
from priml.model.custom_types import ChannelsInOutConfig
from priml.model.norm import BatchNorm2d
from priml.model.squeeze_excitation import SqueezeExcitation
from priml.model.swiglu import sigmoid
from priml.testing.cost import assert_cost_matches_torch


@pytest.mark.parametrize(
    "config",
    [
        CoAtPool.Config(),
        CoAtMBConv.Config(),
        CoAtAttention.Config(),
        CoAtTransformer.Config(),
    ],
)
def test_composite_boundaries_infer_only_unspecified_widths(
    config: Fig[object],
) -> None:
    for channels_in, channels_out in [(3, -1), (-1, 5), (3, 6)]:
        resolved: object = config.copy_tree()
        assert isinstance(resolved, ChannelsInOutConfig)
        resolved.channels_in, resolved.channels_out = channels_in, channels_out
        resolved = resolved.finalize()
        assert resolved.channels_in == (
            channels_out if channels_in == -1 else channels_in
        )
        assert resolved.channels_out == (
            channels_in if channels_out == -1 else channels_out
        )


def kind_blocks[Block: Fig[object]](
    backbone: CoAtNet.Config,
    kind: type[Block],
) -> list[Block]:
    """Return every transition and block config of one kind across the stages."""
    assert isinstance(backbone.stage, list)
    found: list[Block] = []
    for stage in backbone.stage:
        blocks = stage.block if isinstance(stage.block, list) else [stage.block]
        found.extend(
            block for block in [stage.transition, *blocks] if isinstance(block, kind)
        )
    return found


@pytest.mark.parametrize("output_only", [False, True])
def test_inferred_pool_width_uses_the_parameter_free_shortcut(
    output_only: bool,
) -> None:
    config = CoAtPool.Config()
    if output_only:
        config.channels_out = 3
    else:
        config.channels_in = 3
    model = config.make()
    assert isinstance(model.expand, nn.Identity)
    assert list(model.parameters()) == []
    image = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6)
    assert torch.equal(
        model(image),
        functional.avg_pool2d(image, kernel_size=2, stride=2),
    )


@pytest.mark.parametrize("activation", ["silu", "sigmoid", "unsupported"])
def test_activated_norm_cost_prices_the_actual_injected_function(
    activation: str,
) -> None:
    config = CoAtNorm.Config(channels_in=5)
    if activation == "unsupported":
        config.activation = torch.sin
        with pytest.raises(TypeError, match="sin has no cost"):
            cost(
                config.copy_tree().finalize(),
                seq_len=3,
                batch_size=2,
                dtype=torch.float32,
            )
        return
    if activation == "sigmoid":
        config.activation = sigmoid
    norm = BatchNorm2d.Config(channels_in=5, elementwise_affine=True)
    base = cost(norm, seq_len=3, batch_size=2, dtype=torch.float32)
    actual = cost(config, seq_len=3, batch_size=2, dtype=torch.float32)
    primal, adjoint = (5, 5) if activation == "silu" else (4, 3)
    for phase, factor in (("primal", primal), ("adjoint", adjoint)):
        assert (
            actual["flops", phase, "elementwise", torch.float32]
            == base["flops", phase, "elementwise", torch.float32] + factor * 2 * 3 * 5
        )


@pytest.mark.parametrize(
    "component",
    [
        "pool",
        "se",
        "mbconv",
        "attention",
        "scale",
        "mlp",
        "downnorm",
        "transformer",
        "stem",
        "stage",
        "network",
    ],
)
def test_native_cost_counts_every_projection_and_owned_parameter(
    component: str,
) -> None:
    channels, grid = 5, (6, 10)
    if component == "pool":
        config = CoAtPool.Config(channels_in=channels, channels_out=7)
    elif component == "se":
        channels = 12
        config = SqueezeExcitation.Config(
            channels_in=channels,
            channels_hidden=channels // 4,
            act=silu_inplace,
            fc1=Conv2d.Config(kernel_size=1, padding=0, bias=True),
            fc2=Conv2d.Config(kernel_size=1, padding=0, bias=True),
        )
    elif component == "mbconv":
        config = CoAtMBConv.Config(channels_in=channels, channels_out=7, stride=2)
    elif component == "attention":
        channels, grid = 12, (5, 7)
        config = CoAtAttention.Config(
            channels_in=channels,
            channels_out=13,
            channels_head=3,
        )
        config.relative.feature_size = grid
        config.relative.mlp.channels_hidden = 6
    elif component == "scale":
        config = CoAtScale.Config(channels_in=channels)
    elif component == "mlp":
        config = CoAtConvMLP.Config(channels_in=channels)
    elif component == "downnorm":
        config = CoAtDownNorm.Config()
        config.norm.channels_in = channels
        config.down.channels_in = config.down.channels_out = channels
    elif component == "transformer":
        channels = 12
        config = CoAtTransformer.Config(
            channels_in=channels,
            channels_out=13,
            norm1=CoAtDownNorm.Config(),
            shortcut=CoAtPool.Config(),
        )
        config.attention.channels_head = 3
        config.attention.relative.feature_size = (3, 5)
        config.attention.relative.mlp.channels_hidden = 6
    elif component == "stem":
        channels = 3
        config = CoAtStem.Config(channels_hidden=(5, 7))
    elif component == "stage":
        config = CoAtStage.Config(
            channels_in=channels,
            channels_out=7,
            feature_size=(3, 5),
            blocks_per_stage=3,
        )
    else:
        channels, grid = 3, (32, 64)
        config = _config()
        config.image_size = grid
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, channels, *grid, requires_grad=True),
        input_grid=grid,
        batch_size=2,
        dtype=torch.float32,
    )


def test_coordinate_mlp_cost_counts_both_trainable_projections() -> None:
    config = CoAtLinearMLP.Config(channels_hidden=7, channels_out=11)
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(3, 5, 2, requires_grad=True),
        seq_len=5,
        batch_size=3,
        dtype=torch.float32,
    )


def test_se_cost_includes_the_complete_broadcast_gate_gradient() -> None:
    config = SqueezeExcitation.Config(
        channels_in=8,
        channels_hidden=2,
        act=silu_inplace,
        fc1=Conv2d.Config(kernel_size=1, padding=0, bias=True),
        fc2=Conv2d.Config(kernel_size=1, padding=0, bias=True),
    )
    actual = cost(
        config.copy_tree().finalize(),
        input_grid=(4, 5),
        batch_size=3,
        dtype=torch.float32,
    )
    assert actual["flops", "adjoint", "reduction", torch.float32] == (
        (2 + 8) * (3 - 1) + 3 * 8 * (4 * 5 - 1)
    )
    assert (
        actual["bytes", "adjoint", "reduction", torch.float32]
        == ((3 + 1) * (2 + 8) + 3 * 8 * (4 * 5 + 1)) * torch.float32.itemsize
    )
    smaller = cost(
        config.copy_tree().finalize(),
        input_grid=(2, 3),
        batch_size=3,
        dtype=torch.float32,
    )
    for phase, operands in (("primal", 2), ("adjoint", 9)):
        assert (
            actual["bytes", phase, "elementwise", torch.float32]
            - smaller["bytes", phase, "elementwise", torch.float32]
            == operands * 3 * 8 * (4 * 5 - 2 * 3) * torch.float32.itemsize
        )


def test_attention_cost_includes_the_broadcast_bias_application_and_gradient() -> None:
    config = CoAtAttention.Config(channels_in=15, channels_out=7, channels_head=3)
    config.relative.feature_size = (2, 4)
    config.relative.mlp.channels_hidden = 9
    config = config.copy_tree().finalize()
    actual = cost(config, input_grid=(2, 4), batch_size=6, dtype=torch.float32)
    base = (
        cost(config.qkv, input_grid=(2, 4), batch_size=6, dtype=torch.float32)
        + cost(config.relative, batch_size=6, dtype=torch.float32)
        + attention_kernel_cost(
            seq_len=8,
            batch_size=6,
            dtype=torch.float32,
            num_heads=5,
            channels_head=3,
        )
        + cost(config.projection, input_grid=(2, 4), batch_size=6, dtype=torch.float32)
    )
    assert actual["flops", "primal", "elementwise", torch.float32] == (
        base["flops", "primal", "elementwise", torch.float32] + 6 * 5 * 8**2
    )
    assert actual["flops", "adjoint", "reduction", torch.float32] == (
        base["flops", "adjoint", "reduction", torch.float32] + (6 - 1) * 5 * 8**2
    )
    assert actual["bytes", "primal", "elementwise", torch.float32] == (
        base["bytes", "primal", "elementwise", torch.float32]
        + (2 * 6 + 1) * 5 * 8**2 * torch.float32.itemsize
    )
    assert actual["bytes", "adjoint", "reduction", torch.float32] == (
        base["bytes", "adjoint", "reduction", torch.float32]
        + (6 + 1) * 5 * 8**2 * torch.float32.itemsize
    )


def test_relative_bias_cost_excludes_constant_coordinate_gradients() -> None:
    config = CoAtRelativeBias.Config(feature_size=(3, 5), num_heads=7)
    config.mlp.channels_hidden = 11
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.zeros(()),
        run=_bias_forward,
        batch_size=2,
        dtype=torch.float32,
    )


def _bias_forward(module: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    del inputs
    assert isinstance(module, CoAtRelativeBias)
    return module.get_bias()


def _config() -> CoAtNet.Config:
    config = CoAtNet.Config()
    config.image_size = (64, 96)
    config.stem.channels_hidden = (1, 2)
    config.channels_hidden = (2, 4, 2, 4)
    config.blocks_per_stage = (2, 2, 3, 1)
    for block in kind_blocks(config, CoAtTransformer.Config):
        block.attention.channels_head = 2
        block.attention.relative.mlp.channels_hidden = 2
    return config


def test_forward_shape_and_state_dict_layout() -> None:
    model = _config().make().eval()
    assert model(torch.zeros(2, 3, 64, 96)).shape == (2, 4)
    names = list(model.state_dict())
    assert names[0] == "stem.conv1.weight"
    assert names[-2:] == ["norm.weight", "norm.bias"]
    for expected in (
        "stages.0.blocks.0.pre_norm.weight",
        "stages.0.blocks.0.conv2_kxk.weight",
        "stages.0.blocks.0.se.fc1.weight",
        "stages.1.blocks.0.shortcut.expand.weight",
        "stages.2.blocks.0.norm1.norm.weight",
        "stages.2.blocks.0.attn.qkv.weight",
        "stages.2.blocks.0.attn.rel_pos.mlp.fc1.weight",
        "stages.2.blocks.0.ls1.gamma",
        "stages.2.blocks.1.mlp.fc2.weight",
        "stages.2.blocks.1.norm2.weight",
    ):
        assert expected in names
    assert not any("relative_position_index" in name for name in names)


def test_loaded_state_reproduces_the_donor_output() -> None:
    donor = _config().make().eval()
    with torch.no_grad():
        for parameter in donor.parameters():
            parameter.add_(torch.rand_like(parameter) / 10)
    model = _config().make().eval()
    model.load_checkpoint(donor.state_dict())
    image = (
        torch.arange(2 * 3 * 64 * 96, dtype=torch.float32).reshape(2, 3, 64, 96) / 9_000
    )
    assert torch.equal(model(image), donor(image))


def test_coatnet_checkpoint_validation_is_atomic() -> None:
    model = _config().make()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    for bad in (
        dict[str, torch.Tensor](),
        {**state, "norm.weight": torch.zeros(1)},
        {**state, "norm.bias": state["norm.bias"].double()},
    ):
        with pytest.raises(ValueError, match="Checkpoint"):
            model.load_checkpoint(bad)
        assert all(
            torch.equal(value, state[name])
            for name, value in model.state_dict().items()
        )


def test_defaults_are_the_canonical_coatnet_2_layout() -> None:
    config = CoAtNet.Config().copy_tree().finalize()
    assert config.channels_hidden == (128, 256, 512, 1024)
    assert config.blocks_per_stage == (2, 6, 14, 2)
    assert config.image_size == (224, 224)
    assert config.feature_width() == 1024
    assert isinstance(config.stage, list)
    assert [stage.feature_size for stage in config.stage] == [
        (56, 56),
        (28, 28),
        (14, 14),
        (7, 7),
    ]
    assert [stage.blocks_per_stage for stage in config.stage] == [2, 6, 14, 2]
    kinds = [type(stage.transition) for stage in config.stage]
    assert kinds == [CoAtMBConv.Config] * 2 + [CoAtTransformer.Config] * 2


@torch.inference_mode()
def test_configured_stem_stride_derives_every_stage_and_relative_grid() -> None:
    config = _config()
    config.blocks_per_stage = (1, 1, 1, 1)
    config.stem.conv1.stride = 1
    resolved = config.copy_tree().finalize()
    assert isinstance(resolved.stage, list)
    assert [stage.feature_size for stage in resolved.stage] == [
        (32, 48),
        (16, 24),
        (8, 12),
        (4, 6),
    ]
    for stage in resolved.stage[2:]:
        block = stage.transition
        assert isinstance(block, CoAtTransformer.Config)
        assert block.attention.relative.feature_size == stage.feature_size
    output = config.make().eval()(torch.zeros(2, 3, 64, 96))
    assert output.shape == (2, 4)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("transformer", [False, True])
def test_stride_one_blocks_reject_unequal_widths_before_parameter_draws(
    transformer: bool,
) -> None:
    config = (
        CoAtTransformer.Config(channels_in=6, channels_out=7)
        if transformer
        else CoAtMBConv.Config(channels_in=3, channels_out=5)
    )
    if isinstance(config, CoAtTransformer.Config):
        config.attention.channels_head = 3
        config.attention.relative.feature_size = (3, 5)
    with torch.random.fork_rng(devices=[]):
        before = torch.get_rng_state()
        with pytest.raises(ValueError, match=r"channels_in=.*must equal channels_out"):
            config.make()
        assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("stage_index", [0, 2])
def test_network_rejects_stride_one_width_mismatch_before_stem_allocation(
    stage_index: int,
) -> None:
    config = _config().copy_tree().finalize()
    assert isinstance(config.stage, list)
    stage = config.stage[stage_index]
    assert isinstance(stage.block, list)
    block = stage.block[0]
    assert isinstance(block, (CoAtMBConv.Config, CoAtTransformer.Config))
    block.channels_out += 1
    allocations: list[torch.Tensor] = []
    config.stem.conv1.init_weight = allocations.append
    with pytest.raises(ValueError, match=r"channels_in=.*must equal channels_out"):
        config.make()
    assert allocations == []


@pytest.mark.parametrize("image_size", [(80, 112), (65, 96)])
def test_incompatible_pool_and_convolution_grids_fail_before_allocation(
    image_size: tuple[int, int],
) -> None:
    config = _config()
    config.image_size = image_size
    config.blocks_per_stage = (1, 1, 1, 1)
    allocations: list[torch.Tensor] = []
    config.stem.conv1.init_weight = allocations.append
    with pytest.raises(ValueError, match="grid"):
        config.make()
    assert allocations == []


@pytest.mark.parametrize(
    "component",
    [
        "stem",
        "expansion",
        "depthwise",
        "projection",
        "shortcut",
        "qkv",
        "attention",
        "mlp",
    ],
)
def test_grid_validation_uses_configured_component_geometry(component: str) -> None:
    config = _config()
    config.blocks_per_stage = (1, 1, 1, 1)
    if component == "stem":
        config.image_size = (63, 95)
        config.stem.conv1.kernel_size = 2
        config.stem.conv1.padding = 0
    elif component == "expansion":
        for conv_block in kind_blocks(config, CoAtMBConv.Config):
            conv_block.conv1_1x1.stride = 2
    elif component == "depthwise":
        for conv_block in kind_blocks(config, CoAtMBConv.Config):
            conv_block.conv2_kxk.dilation = 2
    elif component == "projection":
        for conv_block in kind_blocks(config, CoAtMBConv.Config):
            conv_block.conv3_1x1.padding = 1
    elif component == "shortcut":
        for conv_block in kind_blocks(config, CoAtMBConv.Config):
            conv_block.shortcut.projection.kernel_size = 2
    elif component == "qkv":
        for attention_block in kind_blocks(config, CoAtTransformer.Config):
            attention_block.attention.qkv.padding = 1
    elif component == "attention":
        for attention_block in kind_blocks(config, CoAtTransformer.Config):
            attention_block.attention.projection.padding = 1
    else:
        for attention_block in kind_blocks(config, CoAtTransformer.Config):
            attention_block.mlp.fc2.padding = 1
    allocations: list[torch.Tensor] = []
    config.stem.conv1.init_weight = allocations.append
    with pytest.raises(ValueError, match="grid"):
        config.make()
    assert allocations == []


def test_explicit_relative_bias_grid_fails_before_stem_allocation() -> None:
    config = _config().copy_tree().finalize()
    assert isinstance(config.stage, list)
    block = config.stage[2].transition
    assert isinstance(block, CoAtTransformer.Config)
    block.attention.relative.feature_size = (3, 5)
    allocations: list[torch.Tensor] = []
    config.stem.conv1.init_weight = allocations.append
    with pytest.raises(
        ValueError,
        match=r"^CoAtNet attention grid differs from its relative bias grid\.$",
    ):
        config.make()
    assert allocations == []


@pytest.mark.parametrize("image_size", [(4, 6), (3, 6)])
def test_nonpositive_convolution_grids_fail_before_stem_allocation(
    image_size: tuple[int, int],
) -> None:
    config = _config()
    config.image_size = image_size
    config.stem.conv1.kernel_size = 5
    config.stem.conv1.stride = 1
    config.stem.conv1.padding = 0
    allocations: list[torch.Tensor] = []
    config.stem.conv1.init_weight = allocations.append
    with pytest.raises(
        ValueError,
        match=r"^CoAtNet convolution grid must remain positive\.$",
    ):
        config.make()
    assert allocations == []


@pytest.mark.parametrize("image_size", [(224, 224), (320, 320), (63, 95)])
@torch.inference_mode()
def test_compatible_and_nondivisible_grids_forward(
    image_size: tuple[int, int],
) -> None:
    config = _config()
    config.image_size = image_size
    config.blocks_per_stage = (1, 1, 1, 1)
    model = config.make().eval()
    # CoAtNet.forward consumes three-channel images by default.
    output = model(torch.zeros(2, 3, *image_size))
    assert output.shape == (2, config.channels_hidden[-1])
    assert torch.isfinite(output).all()


def test_stage_configs_factor_identical_blocks_without_sharing_parameters() -> None:
    config = _config().copy_tree().finalize()
    assert isinstance(config.stage, list)
    assert [1 + len(cast(list[object], stage.block)) for stage in config.stage] == [
        2,
        2,
        3,
        1,
    ]
    model = config.make()
    sizes: list[int] = []
    for stage in model.stages:
        assert isinstance(stage, CoAtStage)
        sizes.append(len(stage.blocks))
    assert sizes == [2, 2, 3, 1]
    stage = model.stages[2]
    assert isinstance(stage, CoAtStage)
    first = dict(stage.blocks.get_submodule("1").named_parameters())
    second = dict(stage.blocks.get_submodule("2").named_parameters())
    assert first.keys() == second.keys()
    assert all(first[name].data_ptr() != second[name].data_ptr() for name in first)


def test_coatnet_gradients_reset_and_invalid_head_geometry() -> None:
    model = _config().make()
    model(torch.randn(2, 3, 64, 96)).sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, CoAtScale):
                module.gamma.fill_(3)
    model.reset_parameters()
    for module in model.modules():
        if isinstance(module, CoAtScale):
            assert torch.equal(module.gamma, torch.full_like(module.gamma, 1e-6))
    config = CoAtAttention.Config()
    config.channels_in, config.channels_out = 3, 4
    config.channels_head = 2
    with pytest.raises(
        ValueError,
        match=r"^Attention width must divide into complete heads\.$",
    ):
        config.make()


def test_mbconv_residual_grid_mismatch_is_named() -> None:
    config = _config()
    config.blocks_per_stage = (1, 1, 1, 1)
    for conv_block in kind_blocks(config, CoAtMBConv.Config):
        conv_block.conv3_1x1.padding = 1
    with pytest.raises(ValueError, match=r"^CoAtNet MBConv residual grids differ\.$"):
        config.make()


def test_stage_requires_its_transition_block() -> None:
    with pytest.raises(ValueError, match="requires its transition block"):
        CoAtStage.Config(blocks_per_stage=0).finalize()


def test_mbconv_rounds_a_small_expansion_up_to_the_next_multiple_of_eight() -> None:
    config = CoAtMBConv.Config(channels_in=11, channels_out=11, expansion=1).finalize()
    assert config.conv1_1x1.channels_out == 16


def test_image_too_small_for_the_pooling_stages_names_the_pool_grid() -> None:
    config = _config()
    config.image_size = (16, 16)
    config.blocks_per_stage = (1, 1, 1, 1)
    with pytest.raises(
        ValueError,
        match=r"^CoAtNet pooling grid must remain positive\.$",
    ):
        config.make()


@pytest.mark.parametrize("component", ["attention", "mlp"])
def test_residual_grid_mismatch_in_a_transformer_block_is_named(component: str) -> None:
    config = _config()
    config.blocks_per_stage = (1, 1, 1, 1)
    for block in kind_blocks(config, CoAtTransformer.Config):
        if component == "attention":
            block.attention.projection.padding = 1
        else:
            block.mlp.fc2.padding = 1
    with pytest.raises(
        ValueError,
        match=r"^CoAtNet attention or MLP residual grids differ\.$",
    ):
        config.make()


def test_unused_shortcut_projection_geometry_is_ignored_when_widths_match() -> None:
    config = _config().copy_tree().finalize()
    assert isinstance(config.stage, list)
    block = config.stage[0].transition
    assert isinstance(block, CoAtMBConv.Config)
    assert block.channels_in == block.channels_out
    block.shortcut.projection.kernel_size = 2
    assert config.make().eval()(torch.zeros(2, 3, 64, 96)).shape == (2, 4)


class _Plain(nn.Module):
    """A normalization stand-in that does not declare how it changes the grid."""

    class Config(Fig["_Plain"]):
        channels_in: int = -1

    def __init__(self, config: Config) -> None:
        super().__init__()
        del config

    def reset_parameters(self) -> None:
        pass

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_a_norm_without_an_output_grid_keeps_the_input_grid() -> None:
    config = _config().copy_tree().finalize()
    assert isinstance(config.stage, list)
    block = config.stage[2].block
    assert isinstance(block, list)
    transformer = block[0]
    assert isinstance(transformer, CoAtTransformer.Config)
    transformer.norm1 = _Plain.Config()
    model = config.make().eval()
    assert model(torch.zeros(2, 3, 64, 96)).shape == (2, 4)


def test_relative_bias_indexes_every_ordered_offset_and_stays_out_of_the_state() -> (
    None
):
    model = CoAtRelativeBias.Config(feature_size=(2, 3), num_heads=2).make()
    expected = [
        (qy - ky + 1) * 5 + (qx - kx + 2)
        for qy in range(2)
        for qx in range(3)
        for ky in range(2)
        for kx in range(3)
    ]
    assert model.relative_position_index.tolist() == expected
    assert model.rel_coords_log.shape == (3, 5, 2)
    for row, dy in enumerate(range(-1, 2)):
        for column, dx in enumerate(range(-2, 3)):
            point = model.rel_coords_log[row, column]
            assert point[0].item() == pytest.approx(
                math.copysign(math.log(1 + abs(dy)), dy) if dy else 0.0,
            )
            assert point[1].item() == pytest.approx(
                math.copysign(math.log(1 + abs(dx)), dx) if dx else 0.0,
            )
    assert not set(dict(model.named_buffers())) & set(model.state_dict())
    assert set(dict(model.named_buffers())) == {
        "relative_position_index",
        "rel_coords_log",
    }


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
