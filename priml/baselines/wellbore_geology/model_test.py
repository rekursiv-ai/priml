"""Check the wellbore networks: spatial blocks, backbone and depth posterior."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Final

import random

from safetensors.torch import save_file
from torch import nn
from torch.nn import functional

import pytest
import torch

from priml.baselines.wellbore_geology.model import (
    AddUp,
    Backbone,
    ConcatUp,
    ConvNormAct,
    ImageNorm,
    PosteriorUNet,
    ResidualBlock,
    SequenceModel,
    backbone_mlp,
    bilinear_cost,
    retained_stage_configs,
)
from priml.baselines.wellbore_geology.testing import tiny_model_config
from priml.cost import cost, map_cost, set_cost, traffic
from priml.model.conv import Conv2d
from priml.model.convnext import ConvNeXtBlock, ConvNeXtStage
from priml.model.custom_types import ChannelsInOutConfig
from priml.model.linear import Linear
from priml.model.norm import BatchNorm, BatchNorm2d, LayerNorm, LayerNorm2d
from priml.model.pool import AvgPool2d
from priml.model.sequential import Sequential
from priml.model.special import Activation, Dropout
from priml.model.swiglu import SwiGLU, gelu, relu, silu
from priml.testing.bfb import host_agnostic_numerics, portable_half_precision
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_tensor_golden


if TYPE_CHECKING:
    from collections.abc import Generator

    from configgle import Fig

    from priml.math.custom_types import TensorFn


_CWD: Final = Path(__file__).resolve().parent


@set_cost(map_cost(primal=1, adjoint=1))
def negative(value: torch.Tensor) -> torch.Tensor:
    """Negate a tensor with the shared elementwise cost convention."""
    return torch.neg(value)


@pytest.mark.parametrize(
    "config",
    [Activation.Config(), Dropout.Config(p=0.0), ImageNorm.Config()],
)
def test_preserving_adapters_reject_contradictory_widths(
    config: Activation.Config | Dropout.Config | ImageNorm.Config,
) -> None:
    config = config.copy_tree()
    config.channels_in = 3
    config.channels_out = 5
    with pytest.raises(ValueError, match="must equal"):
        config.make()


@pytest.mark.parametrize(
    "config",
    [Activation.Config(), Dropout.Config(p=0.0), ImageNorm.Config()],
)
def test_preserving_adapters_declare_spatial_geometry(
    config: Activation.Config | Dropout.Config | ImageNorm.Config,
) -> None:
    config = config.copy_tree()
    config.channels_in = 3
    chain = Sequential.Config(elements=[config])
    assert chain.finalized().output_grid((5, 7)) == (5, 7)
    assert_cost_matches_torch(
        chain,
        build_input=lambda: torch.randn(2, 3, 5, 7, requires_grad=True),
        input_grid=(5, 7),
        seq_len=5 * 7,
        batch_size=2,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("kind", ["activation", "dropout", "norm", "conv", "residual"])
def test_spatial_composition_costs_match_whole_torch_invocations(kind: str) -> None:
    config = {
        "activation": Activation.Config(),
        "dropout": Dropout.Config(p=0.2),
        "norm": ImageNorm.Config(channels_in=3),
        "conv": ConvNormAct.Config(channels_in=3, channels_out=7),
        "residual": ResidualBlock.Config(
            channels_in=3,
            channels_out=7,
            skip=Conv2d.Config(kernel_size=1),
        ),
    }[kind]
    execution = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 5, 11, requires_grad=True),
        input_grid=(5, 11),
        seq_len=55,
        channels=3,
        batch_size=2,
        dtype=torch.float32,
    )
    assert execution["flops", "primal", "elementwise"].sum() > 0
    if kind == "norm":
        assert execution.bytes_state == 2 * 3 * 4 + 8


@pytest.mark.parametrize("concatenate", [False, True])
def test_decoder_costs_follow_rectangular_skip_geometry(concatenate: bool) -> None:
    config = (
        ConcatUp.Config(channels_in=3, channels_skip=7, channels_out=7)
        if concatenate
        else AddUp.Config(channels_in=3)
    )
    assert_cost_matches_torch(
        config,
        build_input=lambda: (
            torch.randn(2, 3, 5, 11, requires_grad=True),
            torch.randn(2, 7 if concatenate else 3, 13, 23, requires_grad=True),
        ),
        input_grid=(5, 11),
        output_grid=(13, 23),
        batch_size=2,
        dtype=torch.float32,
    )


@pytest.mark.parametrize(
    ("activation", "primal", "adjoint"),
    [
        (silu, 5, 5),
        (gelu, 5, 11),
        (negative, 1, 1),
        (relu, 1, 1),
    ],
)
def test_activation_cost_uses_the_shared_elementwise_convention(
    activation: TensorFn,
    primal: int,
    adjoint: int,
) -> None:
    execution = cost(
        Activation.Config(act=activation),
        channels=3,
        seq_len=5,
        batch_size=2,
        dtype=torch.float32,
    )
    assert execution["flops", "primal"].sum() == primal * 30
    assert execution["flops", "adjoint"].sum() == adjoint * 30
    assert execution["bytes", "primal"].sum() == 2 * 30 * 4
    assert execution["bytes", "adjoint"].sum() == 3 * 30 * 4


@pytest.mark.parametrize("probability", [0.0, 0.25, 1.0])
def test_dropout_cost_distinguishes_identity_random_masks_and_complete_drop(
    probability: float,
) -> None:
    execution = cost(
        Dropout.Config(p=probability),
        channels=3,
        seq_len=5,
        batch_size=2,
        dtype=torch.float32,
    )
    assert execution["flops", "primal"].sum() == (
        0 if probability == 0 else 30 if probability == 1 else 60
    )
    assert execution["flops", "adjoint"].sum() == (0 if probability == 0 else 30)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_narrow_image_norm_cost_keeps_float32_statistics_and_casts(
    dtype: torch.dtype,
) -> None:
    config = ImageNorm.Config(channels_in=3).finalize()
    execution = cost(config, seq_len=35, batch_size=2, dtype=dtype)
    full = cost(config, seq_len=35, batch_size=2, dtype=torch.float32)
    for phase in ("primal", "adjoint"):
        assert execution["bytes", phase, dtype].sum() == 2 * 210 * dtype.itemsize
        assert (
            execution["bytes", phase, torch.float32].sum()
            == full["bytes", phase].sum() + 2 * 210 * 4
        )


def test_uncosted_activation_is_rejected() -> None:
    with pytest.raises(TypeError, match="has no cost"):
        cost(
            Activation.Config(act=torch.sin),
            channels=3,
            seq_len=5,
            batch_size=2,
            dtype=torch.float32,
        )


@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("training", [False, True])
def test_image_norm_preserves_layout_and_float32_statistics(
    channels_last: bool,
    dtype: torch.dtype,
    training: bool,
) -> None:
    config = ImageNorm.Config()
    config.channels_in = 3
    config.channels_last = channels_last
    config.bn.eps = 0.003
    config.bn.momentum = 0.2
    layer = config.make().train(training)
    layer.bn.weight.data.copy_(torch.tensor([0.3, 0.7, 1.2]))
    layer.bn.bias.data.copy_(torch.tensor([-0.1, 0.2, 0.5]))
    value = torch.linspace(-2, 3, 2 * 3 * 4 * 5).reshape(2, 3, 4, 5).to(dtype)
    assert layer.bn.running_mean is not None
    assert layer.bn.running_var is not None
    assert layer.bn.num_batches_tracked is not None
    mean = layer.bn.running_mean.clone()
    variance = layer.bn.running_var.clone()
    expected = functional.batch_norm(
        value.float() if dtype != torch.float32 else value,
        mean,
        variance,
        layer.bn.weight,
        layer.bn.bias,
        training=training,
        momentum=0.2,
        eps=0.003,
    ).to(dtype)
    if channels_last:
        value = value.permute(0, 2, 3, 1)
        expected = expected.permute(0, 2, 3, 1)
    actual = layer(value)
    assert actual.dtype == dtype
    assert torch.equal(actual, expected)
    assert torch.equal(layer.bn.running_mean, mean)
    assert torch.equal(layer.bn.running_var, variance)
    assert int(layer.bn.num_batches_tracked) == int(training)
    layer.reset_parameters()
    assert torch.equal(layer.bn.weight, torch.ones(3))
    assert torch.equal(layer.bn.bias, torch.zeros(3))
    assert torch.equal(layer.bn.running_mean, torch.zeros(3))
    assert torch.equal(layer.bn.running_var, torch.ones(3))
    assert int(layer.bn.num_batches_tracked) == 0


@pytest.mark.parametrize("convolution", [False, True])
@pytest.mark.parametrize("custom_bias", [False, True])
def test_fan_in_initialization_matches_torch_constructor_draws(
    convolution: bool,
    custom_bias: bool,
) -> None:
    config = Conv2d.Config() if convolution else Linear.Config()
    config.init_bias = None
    config.channels_in = 3
    config.channels_out = 4
    config.bias = True
    if custom_bias:
        config.init_bias = nn.init.zeros_
    torch.manual_seed(37)
    actual = config.make()
    actual_rng = torch.get_rng_state()
    torch.manual_seed(37)
    expected = nn.Conv2d(3, 4, 3, padding=1) if convolution else nn.Linear(3, 4)
    assert torch.equal(actual.weight, expected.weight)
    assert actual.bias is not None
    assert expected.bias is not None
    if custom_bias:
        assert torch.equal(actual.bias, torch.zeros(4))
    else:
        assert torch.equal(actual.bias, expected.bias)
        assert torch.equal(actual_rng, torch.get_rng_state())


def test_injected_activation_and_dropout_follow_module_contract() -> None:
    config = Activation.Config()
    config.act = negative
    activation = config.make()
    dropout = Dropout.Config(p=0.0).make()
    value = torch.arange(2 * 3).reshape(2, 3).float().requires_grad_()
    actual = dropout(activation(value))
    assert torch.equal(actual, -value)
    actual.sum().backward()
    assert value.grad is not None
    assert torch.equal(value.grad, -torch.ones_like(value))
    activation.reset_parameters()
    dropout.reset_parameters()


def test_explicit_residual_skip_is_preserved() -> None:
    config = ResidualBlock.Config()
    config.channels_in = config.channels_out = 3
    config.activation = negative
    config.skip = Activation.Config(act=negative)
    layer = config.make().eval()
    value = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).float() / 17
    expected = -(layer.drop(layer.bn2(layer.conv2(layer.conv1(value)))) - value)
    assert torch.equal(layer(value), expected)
    assert isinstance(layer.skip, Activation)
    assert layer.skip.act is negative


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_narrow_norm_state_preserves_leaf_values_gradients_and_buffers(
    configured: bool,
    channels_last: bool,
    training: bool,
) -> None:
    config = ImageNorm.Config(channels_in=3)
    config.channels_last = channels_last
    if configured:
        config.bn.dtype = torch.bfloat16
    layer = config.make().train(training)
    if not configured:
        layer.bfloat16()
    reference = deepcopy(layer.bn)
    value = torch.linspace(-2, 3, 120).reshape(2, 3, 4, 5).bfloat16().requires_grad_()
    other = value.detach().clone().requires_grad_()
    expected = reference(other)
    actual = layer(value.permute(0, 2, 3, 1) if channels_last else value)
    assert actual.dtype == torch.bfloat16
    assert torch.equal(
        actual,
        expected.permute(0, 2, 3, 1) if channels_last else expected,
    )
    actual.sum().backward()
    expected.sum().backward()
    assert value.grad is not None
    assert other.grad is not None
    assert torch.equal(value.grad, other.grad)
    for name, state in layer.bn.state_dict().items():
        assert torch.equal(state, reference.state_dict()[name])
    assert layer.bn.weight.grad is not None
    assert reference.weight.grad is not None
    assert torch.equal(layer.bn.weight.grad, reference.weight.grad)
    assert layer.bn.bias.grad is not None
    assert reference.bias.grad is not None
    assert torch.equal(layer.bn.bias.grad, reference.bias.grad)
    assert layer.bn.running_mean is not None
    assert layer.bn.running_mean.dtype == torch.bfloat16


def test_narrow_norm_cost_uses_configured_state_without_spurious_casts() -> None:
    config = ImageNorm.Config(channels_in=3)
    config.bn.dtype = torch.bfloat16
    execution = cost(config.finalized(), seq_len=20, batch_size=2, dtype=torch.bfloat16)
    leaf = cost(config.finalized().bn, seq_len=20, batch_size=2, dtype=torch.bfloat16)
    for phase in ("primal", "adjoint"):
        assert execution["bytes", phase].sum() == leaf["bytes", phase].sum()
        assert execution["bytes", phase, torch.float32].sum() == 0
    assert execution.bytes_state == 2 * 3 * 2 + 8


@pytest.mark.parametrize("entry", ["forward", "cost"])
def test_decoder_residual_rejects_singleton_grid_broadcast(entry: str) -> None:
    config = ResidualBlock.Config(channels_in=3)
    config.conv1.conv.kernel_size = 4
    config.conv1.conv.stride = 4
    config.conv1.conv.padding = 0
    if entry == "forward":
        with pytest.raises(ValueError, match=r"residual.*grid"):
            config.make()(torch.randn(2, 3, 4, 5))
    else:
        with pytest.raises(ValueError, match=r"residual.*grid"):
            cost(
                config.finalized(),
                input_grid=(4, 5),
                batch_size=2,
                dtype=torch.float32,
            )


@pytest.mark.parametrize("nested", [False, True])
def test_decoder_residual_accepts_matching_spatial_projection(nested: bool) -> None:
    config = ResidualBlock.Config(channels_in=3)
    config.conv1.conv.stride = 2
    config.conv1.conv.padding = 1
    config.skip = Conv2d.Config(kernel_size=1, stride=2, padding=0)
    if nested:
        config.skip = Sequential.Config(elements=[config.skip])
    resolved = config.finalized()
    value = torch.randn(2, 3, 4, 6)
    assert resolved.output_grid((4, 6)) == resolved.make()(value).shape[-2:]
    assert_cost_matches_torch(
        config,
        build_input=lambda: value.clone().requires_grad_(),
        input_grid=(4, 6),
        batch_size=2,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("axis", ["batch", "channels"])
def test_additive_skip_rejects_broadcastable_branch_shapes(axis: str) -> None:
    layer = AddUp.Config(channels_in=3).make().eval()
    with pytest.raises(ValueError, match=r"Additive skip.*shapes"):
        # Singleton batch/channel axes are the invalid broadcasting cases.
        layer(
            torch.randn(1 if axis == "batch" else 2, 3, 4, 5),
            torch.randn(2, 1 if axis == "channels" else 3, 6, 7),
        )


@pytest.mark.parametrize("dropout", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_transparent_adapters_preserve_composed_width_and_cost(
    dropout: bool,
    nested: bool,
) -> None:
    adapter = Dropout.Config(p=0.0) if dropout else Activation.Config()
    middle = Sequential.Config(elements=[adapter]) if nested else adapter
    config = Sequential.Config(channels_in=2, channels_out=1)
    config.elements = [Linear.Config(channels_out=3), middle, Linear.Config()]
    layer = config.make()
    assert layer(torch.randn(5, 2)).shape == (5, 1)
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(5, 2, requires_grad=True),
        seq_len=1,
        batch_size=5,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("convolution", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_fan_in_configs_have_finalized_value_equality(
    convolution: bool,
    override: bool,
) -> None:
    config = Conv2d.Config() if convolution else Linear.Config()
    config.init_bias = None
    config.channels_in, config.channels_out = 3, 5
    config.bias = True
    if override:
        config.init_bias = nn.init.ones_
    first, second = config.finalized(), config.finalized()
    assert first == second
    assert first == type(config).deserialize(first.serialize())
    assert config == config.copy_tree()


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_zero_input_linear_preserves_torch_output_and_constructor_rng(
    bias: bool,
    override: bool,
) -> None:
    config = Linear.Config(channels_in=0, channels_out=5, bias=bias, init_bias=None)
    if override:
        config.init_bias = nn.init.ones_
    torch.manual_seed(37)
    initial_rng = torch.get_rng_state()
    with pytest.warns(UserWarning, match="zero-element tensors"):
        actual = config.make()
    actual_rng = torch.get_rng_state()
    torch.manual_seed(37)
    with pytest.warns(UserWarning, match="zero-element tensors"):
        expected = nn.Linear(0, 5, bias=bias)
    expected_rng = torch.get_rng_state()
    if override and expected.bias is not None:
        nn.init.ones_(expected.bias)
    # Linear.__init__ explicitly supports zero input features.
    value = torch.empty(2, 0)
    assert torch.equal(actual(value), expected(value))
    assert actual(value).shape == (2, 5)
    assert torch.equal(actual_rng, initial_rng if override else expected_rng)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_zero_input_convolution_keeps_torch_constructor_draw_policy(
    bias: bool,
    override: bool,
) -> None:
    config = Conv2d.Config(channels_in=0, channels_out=5, bias=bias, init_bias=None)
    if override:
        config.init_bias = nn.init.ones_
    torch.manual_seed(37)
    with pytest.warns(UserWarning, match="zero-element tensors"):
        expected = nn.Conv2d(0, 5, 3, padding=1, bias=bias)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(37)
    with pytest.warns(UserWarning, match="zero-element tensors"):
        actual = config.make()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert actual.weight.shape == expected.weight.shape
    if override and actual.bias is not None:
        assert torch.equal(actual.bias, torch.ones(5))
    # Conv2d's zero-input branch writes no output values or default bias values.
    value = torch.empty(2, 0, 4, 6)
    assert actual(value).shape == expected(value).shape


def test_bilinear_cost_counts_weighted_taps_and_scatter_add_gradients() -> None:
    execution = bilinear_cost(
        input_grid=(3, 5),
        output_grid=(4, 6),
        channels=2,
        batch_size=3,
        dtype=torch.bfloat16,
    )
    outputs, inputs = 3 * 2 * 24, 3 * 2 * 15
    expected = traffic(
        "primal",
        "selection",
        elements=5 * outputs,
        flops=7 * outputs,
        dtype=torch.bfloat16,
    ) + traffic(
        "adjoint",
        "selection",
        elements=9 * outputs + inputs,
        flops=8 * outputs,
        dtype=torch.bfloat16,
    )
    assert execution == expected
    assert execution["flops", "primal"].sum() == 7 * outputs
    assert execution["flops", "adjoint"].sum() == 8 * outputs
    assert execution["bytes", "primal", torch.bfloat16].sum() == 5 * outputs * 2
    assert (
        execution["bytes", "adjoint", torch.bfloat16].sum()
        == (9 * outputs + inputs) * 2
    )


def test_narrow_image_norm_requires_running_statistics() -> None:
    norm = ImageNorm.Config(channels_in=3).make()
    norm.bn.running_mean = None
    with pytest.raises(ValueError, match="running_mean"):
        norm(torch.zeros(2, 3, 4, 5, dtype=torch.float16))


def test_source_backbone_uses_shared_leaves_at_small_depths() -> None:
    config = Backbone.Config().network
    assert config.blocks_per_stage == (3, 3, 27, 3)
    assert type(config.stem.conv) is Conv2d.Config
    assert isinstance(config.stage, ConvNeXtStage.Config)
    assert isinstance(config.stage.downsample, Sequential.Config)
    assert isinstance(config.stage.downsample.elements, list)
    assert type(config.stage.downsample.elements[1]) is Conv2d.Config
    assert isinstance(config.stage.block, ConvNeXtBlock.Config)
    assert type(config.stage.block.conv_dw) is Conv2d.Config
    assert type(config.stage.block.mlp) is SwiGLU.Config


def test_feedforward_cost_matches_both_complete_affine_projections() -> None:
    assert_cost_matches_torch(
        backbone_mlp(channels_in=3, channels_out=7),
        build_input=lambda: torch.randn(2, 5, 11, 3, requires_grad=True),
        seq_len=55,
        batch_size=2,
        dtype=torch.float32,
    )


def test_cost_stages_match_native_buffers_without_mutation() -> None:
    config = Backbone.Config()
    config.network.channels_hidden = (2, 3)
    config.network.blocks_per_stage = (1, 1)
    config = config.finalize()
    before = repr(config)
    native = config.make()
    stages = retained_stage_configs(config)
    assert repr(config) == before
    assert len(stages) == len(native.stages) == 2
    grid = (5, 11)
    for index, (stage, actual) in enumerate(zip(stages, native.stages, strict=True)):
        assert isinstance(stage.block, list)
        assert isinstance(stage.block[0].norm, ImageNorm.Config)
        assert stage.block[0].norm.channels_last
        if index:
            assert isinstance(stage.downsample, Sequential.Config)
            assert isinstance(stage.downsample.elements, list)
            norm = stage.downsample.elements[0]
            assert isinstance(norm, ImageNorm.Config)
            assert not norm.channels_last
        execution = cost(stage, input_grid=grid, batch_size=2, dtype=torch.float32)
        assert execution.params == sum(
            parameter.numel() for parameter in actual.parameters()
        )
        assert execution.bytes_state == sum(
            buffer.numel() * buffer.element_size() for buffer in actual.buffers()
        )
        grid = tuple(actual(torch.randn(2, stage.channels_in, *grid)).shape[-2:])


def test_complete_pretrained_state_loads_before_norm_conversion(tmp_path: Path) -> None:
    config = Backbone.Config()
    config.network.channels_hidden = (2, 3, 4, 5)
    config.network.blocks_per_stage = (1, 1, 1, 1)
    stage = config.network.stage
    assert isinstance(stage, ConvNeXtStage.Config)
    assert isinstance(stage.block, ConvNeXtBlock.Config)
    stage.block.conv_dw.kernel_size = 3
    stage.block.conv_dw.padding = 1
    stage.block = [stage.block]
    config.network.stage = [stage.copy_tree() for _ in range(4)]
    raw = config.copy_tree().finalize().network.make()
    with torch.no_grad():
        for index, parameter in enumerate(raw.parameters()):
            parameter.fill_(index / 100)
    original = raw.state_dict()
    state = {
        name.replace("stem.conv.", "stem.0.")
        .replace("stem.norm.", "stem.1.")
        .replace(".mlp.up_proj.", ".mlp.fc1.")
        .replace(".mlp.down_proj.", ".mlp.fc2."): value.contiguous()
        for name, value in original.items()
    }
    state["head.fc.weight"] = torch.zeros(3, 5)
    state["head.fc.bias"] = torch.zeros(3)
    config.checkpoint = tmp_path / "weights.safetensors"
    save_file(state, str(config.checkpoint))
    torch.manual_seed(27)
    loaded = config.make()
    after_load = torch.get_rng_state()
    config.checkpoint = None
    torch.manual_seed(27)
    config.make()
    assert torch.equal(torch.get_rng_state(), after_load)
    actual = loaded.stages.state_dict()
    for name, value in original.items():
        if not name.startswith("stages."):
            continue
        target = (
            name.removeprefix("stages.")
            .replace(".norm.", ".norm.bn.")
            .replace(".downsample.0.", ".downsample.0.bn.")
        )
        assert torch.equal(actual[target], value), name
    feedforward = loaded.stages[0].get_submodule("blocks.0.mlp")
    assert isinstance(feedforward, SwiGLU)
    before = feedforward.up_proj.weight.detach().clone()
    feedforward.reset_parameters()
    assert not torch.equal(feedforward.up_proj.weight, before)


def test_non_affine_backbone_norm_fails_before_replacement() -> None:
    config = Backbone.Config()
    config.network.channels_hidden = (2, 3)
    config.network.blocks_per_stage = (1, 1)
    stage = ConvNeXtStage.Config()
    assert isinstance(stage.block, ConvNeXtBlock.Config)
    stage.block.norm = LayerNorm.Config(elementwise_affine=False)
    config.network.stage = stage
    with pytest.raises(ValueError, match="must be affine"):
        config.make()


def test_backbone_keeps_injected_leaf_configuration() -> None:
    config = Backbone.Config()
    config.network.channels_hidden = (4, 5)
    config.network.blocks_per_stage = (1, 1)
    config.network.stem.conv.dtype = torch.float64
    config.network.stem.conv.device = "cpu"
    assert isinstance(config.network.stage, ConvNeXtStage.Config)
    assert isinstance(config.network.stage.downsample, Sequential.Config)
    assert isinstance(config.network.stage.downsample.elements, list)
    projection = config.network.stage.downsample.elements[1]
    assert isinstance(projection, Conv2d.Config)
    projection.dtype = torch.float64
    assert isinstance(config.network.stage.block, ConvNeXtBlock.Config)
    config.network.stage.block.conv_dw.kernel_size = 3
    config.network.stage.block.conv_dw.padding = 1
    config.network.stage.block.mlp = Activation.Config(act=torch.neg)
    resolved = config.finalized()
    assert resolved.network.stem.conv.dtype == torch.float64
    assert resolved.network.stem.conv.device == "cpu"
    assert isinstance(resolved.network.stage, list)
    for stage in resolved.network.stage:
        if isinstance(stage.downsample, Sequential.Config):
            assert isinstance(stage.downsample.elements, list)
            projection = stage.downsample.elements[1]
            assert isinstance(projection, Conv2d.Config)
            assert projection.dtype == torch.float64
        assert isinstance(stage.block, list)
        for block in stage.block:
            assert block.conv_dw.kernel_size == 3
            assert block.conv_dw.padding == 1
            assert isinstance(block.mlp, Activation.Config)
            assert block.mlp.act is torch.neg


def test_backbone_declares_source_children_before_finalization() -> None:
    config = Backbone.Config()
    assert isinstance(config.network.stem.conv, Conv2d.Config)
    assert isinstance(config.network.stage, ConvNeXtStage.Config)
    assert isinstance(config.network.stage.downsample, Sequential.Config)
    assert isinstance(config.network.stage.downsample.elements, list)
    assert isinstance(config.network.stage.downsample.elements[1], Conv2d.Config)
    assert isinstance(config.network.stage.block, ConvNeXtBlock.Config)
    assert isinstance(config.network.stage.block.conv_dw, Conv2d.Config)
    assert isinstance(config.network.stage.block.mlp, SwiGLU.Config)


def test_feedforward_preserves_injected_affine_bias_and_precision() -> None:
    config = backbone_mlp(channels_in=3, channels_out=5)
    assert isinstance(config.up_proj, Linear.Config)
    assert isinstance(config.down_proj, Linear.Config)
    config.up_proj.bias = config.down_proj.bias = False
    config.up_proj.dtype = config.down_proj.dtype = torch.float64
    layer = config.make()
    assert layer.up_proj.bias is None
    assert layer.down_proj.bias is None
    assert layer.up_proj.weight.dtype == torch.float64
    assert layer.down_proj.weight.dtype == torch.float64
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, dtype=torch.float64, requires_grad=True),
        seq_len=1,
        batch_size=2,
        dtype=torch.float64,
    )


def test_backbone_normalization_conversion_preserves_configured_precision() -> None:
    config = Backbone.Config()
    config.network.channels_hidden = (4, 5)
    config.network.blocks_per_stage = (1, 1)
    stage = config.network.stage
    assert isinstance(stage, ConvNeXtStage.Config)
    assert isinstance(stage.downsample, Sequential.Config)
    assert isinstance(stage.downsample.elements, list)
    stage.downsample.elements[0] = LayerNorm2d.Config(dtype=torch.float64, device="cpu")
    assert isinstance(stage.block, ConvNeXtBlock.Config)
    stage.block.norm = LayerNorm.Config(
        elementwise_affine=True,
        dtype=torch.float64,
        device="cpu",
    )
    resolved = config.finalized()
    native = resolved.make()
    for declared, actual in zip(
        retained_stage_configs(resolved),
        native.stages,
        strict=True,
    ):
        assert isinstance(declared.block, list)
        for block in declared.block:
            assert isinstance(block.norm, ImageNorm.Config)
            assert block.norm.bn.device == "cpu"
        for module in actual.modules():
            if isinstance(module, ImageNorm):
                assert module.bn.weight.dtype == torch.float64
                assert module.bn.running_mean is not None
                assert module.bn.running_mean.dtype == torch.float64
        execution = cost(declared, input_grid=(5, 7), batch_size=2, dtype=torch.float32)
        assert execution.bytes_state == sum(
            value.numel() * value.element_size() for value in actual.buffers()
        )


@pytest.fixture(autouse=True)
def native_cpu_convolutions() -> Generator[None]:
    """Count complete CPU adjoints through Torch's portable convolution kernels."""
    # Keep strided-stem cost witnesses on the portable CPU convolution path.
    with portable_half_precision():
        yield


@pytest.mark.parametrize(
    "config",
    [
        backbone_mlp(),
        ConvNormAct.Config(),
        ResidualBlock.Config(skip=Conv2d.Config(kernel_size=1)),
        ConcatUp.Config(channels_skip=2),
        PosteriorUNet.Config(
            channels_hidden=(2, 3, 4, 5),
        ),
    ],
)
@pytest.mark.parametrize(("channels_in", "channels_out"), [(3, -1), (-1, 5), (3, 7)])
def test_projection_configs_infer_only_missing_boundary_widths(
    config: Fig[object],
    channels_in: int,
    channels_out: int,
) -> None:
    copied: object = config.copy_tree()
    assert isinstance(copied, ChannelsInOutConfig)
    copied.channels_in, copied.channels_out = channels_in, channels_out
    resolved = copied.finalize()
    assert resolved.channels_in == (channels_out if channels_in == -1 else channels_in)
    assert resolved.channels_out == (
        channels_in if channels_out == -1 else channels_out
    )


@pytest.mark.parametrize(("channels_in", "channels_out"), [(3, -1), (-1, 3)])
def test_image_normalization_infers_either_width(
    channels_in: int,
    channels_out: int,
) -> None:
    resolved = ImageNorm.Config(
        channels_in=channels_in,
        channels_out=channels_out,
    ).finalize()
    assert resolved.channels_in == resolved.channels_out == resolved.bn.channels_in == 3


def test_image_normalization_rejects_an_explicit_width_change() -> None:
    with pytest.raises(ValueError, match="must equal"):
        ImageNorm.Config(channels_in=3, channels_out=5).make()


def test_injected_residual_projection_inherits_both_boundary_widths() -> None:
    config = ResidualBlock.Config(channels_in=3, channels_out=5)
    config.skip = Conv2d.Config(kernel_size=1)
    resolved = config.copy_tree().finalize()
    assert isinstance(resolved.skip, Conv2d.Config)
    assert resolved.skip.channels_in == 3
    assert resolved.skip.channels_out == 5
    block = config.make().eval()
    assert block(torch.randn(2, 3, 4, 7)).shape == (2, 5, 4, 7)


@pytest.mark.parametrize("grid", [(5, 7), (6, 8), (7, 11)])
@pytest.mark.parametrize("kernel", [(2, 4), (3, 2)])
@pytest.mark.parametrize("stride", [(3, 5), (2, 3)])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("ceil", [False, True])
def test_pool_geometry_matches_torch_partial_and_padded_windows(
    grid: tuple[int, int],
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: int,
    ceil: bool,
) -> None:
    pool = AvgPool2d.Config(
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        ceil_mode=ceil,
    )
    actual = pool.make()(torch.randn(2, 3, *grid))
    assert pool.output_grid(grid) == actual.shape[-2:]


def test_pool_geometry_rejects_a_missing_spatial_axis() -> None:
    with pytest.raises(ValueError, match="two spatial axes"):
        AvgPool2d.Config(kernel_size=(2, 4)).output_grid((5,))


@pytest.mark.parametrize("heads", [False, True])
def test_complete_model_cost_counts_retained_stages_and_all_heads(heads: bool) -> None:
    config = tiny_model_config(dropout=0)
    finalized = config.copy_tree().finalize()
    chosen = finalized if heads else finalized.unet
    execution = assert_cost_matches_torch(
        chosen,
        build_input=lambda: torch.randn(2, 3, 16, 32, requires_grad=True),
        run=_all_cost_outputs,
        input_grid=(16, 32),
        batch_size=2,
        dtype=torch.float32,
    )
    model = chosen.make()
    assert execution.bytes_state == sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )


def test_whole_fresh_constructor_and_rng() -> None:
    # The nexus family keeps each decoder site's own dropout.
    config = tiny_model_config(dropout=-1)
    torch.manual_seed(27)
    random.seed(31)
    with host_agnostic_numerics():
        model = config.make()
        actual = {name: value.clone() for name, value in model.state_dict().items()}
        actual.update(
            {f"buffer/{name}": value.clone() for name, value in model.named_buffers()},
        )
        actual["rng/torch"] = torch.get_rng_state()
        actual["rng/python/version"] = torch.tensor(random.getstate()[0])
        actual["rng/python/keys"] = torch.tensor(
            random.getstate()[1],
            dtype=torch.int64,
        )
    assert_tensor_golden(_CWD / "testdata" / "constructor.pt", actual)


@pytest.mark.parametrize("dropout", [0.0, 0.05])
def test_tiny_model_keeps_every_stage_and_the_posterior_mean(dropout: float) -> None:
    config = tiny_model_config(dropout=dropout)
    with host_agnostic_numerics():
        model = config.make()
        assert sum(parameter.numel() for parameter in model.parameters()) == 1_121
        assert isinstance(model.get_submodule("proj_out"), Conv2d)
        assert isinstance(model.get_submodule("unet_gr_rmse_head.0"), Linear)
        assert isinstance(model.get_submodule("unet.stem.bn2"), BatchNorm2d)
        # The train-step records fix this complete two-well geometry.
        value = (torch.arange(2 * 3 * 16 * 32).reshape(2, 3, 16, 32) % 17).float() / 19
        prediction, extra = model(value)
        assert prediction.shape == (2, 16)
        assert extra["alignment_logits"].shape == (2, 16, 32)
        assert extra["unet_GR_RMSE_pred"].shape == prediction.shape
        assert extra["regression_logits"] is extra["alignment_logits"]
        assert config.target_stats is not None
        mean, std = config.target_stats
        axis = (torch.linspace(-100 + 100 / 32, 100 - 100 / 32, 32) - mean) / std
        # SequenceModel.forward broadcasts the fixed typewell axis across wells/rows.
        assert torch.equal(
            prediction,
            (extra["alignment_logits"].softmax(-1) * axis.reshape(1, 1, -1)).sum(-1),
        )
        prediction.sum().backward()
        assert model.get_parameter("proj_out.weight").grad is not None


def test_supplied_auxiliary_head_inherits_model_boundaries() -> None:
    config = SequenceModel.Config()
    config.channels_hidden = 4
    config.unet_gr_rmse_head.elements = [Linear.Config(channels_out=1)]
    head = config.finalized().unet_gr_rmse_head
    assert head.channels_in == 4
    assert head.channels_out == 1
    assert head.make()(torch.randn(6, 4)).shape == (6, 1)


def test_child_architecture_is_the_single_model_owner() -> None:
    config = SequenceModel.Config()
    config.unet.channels_hidden = (4, 5)
    config.unet.up_blocks = [ConcatUp.Config()]
    config.unet.backbone.network.blocks_per_stage = (1, 1)
    stage = config.unet.backbone.network.stage
    assert isinstance(stage, ConvNeXtStage.Config)
    assert isinstance(stage.block, ConvNeXtBlock.Config)
    stage.block.conv_dw.kernel_size = 3
    stage.block.conv_dw.padding = 1
    resolved = config.finalized()
    assert resolved.unet.channels_hidden == (4, 5)
    assert resolved.unet.backbone.network.blocks_per_stage == (1, 1)
    assert isinstance(resolved.unet.backbone.network.stage, list)
    for child in resolved.unet.backbone.network.stage:
        assert isinstance(child.block, list)
        assert child.block[0].conv_dw.kernel_size == 3


@pytest.mark.parametrize("stride", [1, 2])
def test_supplied_decoder_inherits_boundaries_and_costs_its_output(stride: int) -> None:
    config = PosteriorUNet.Config(channels_in=3, channels_out=4)
    config.channels_hidden = (4, 5)
    config.backbone.network.blocks_per_stage = (1, 1)
    up = ConcatUp.Config()
    up.proj.stride = stride
    up.proj.padding = 0
    config.up_blocks = [up]
    resolved = config.finalized()
    assert resolved.up_blocks[0].channels_in == 5
    assert resolved.up_blocks[0].channels_skip == 4
    assert resolved.up_blocks[0].channels_out == 4
    assert resolved.up_blocks[0].proj.stride == stride
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 6, 20, requires_grad=True),
        input_grid=(6, 20),
        batch_size=2,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("stride", [1, 2])
def test_unet_cost_starts_at_executed_stem_grid(stride: int) -> None:
    config = PosteriorUNet.Config(channels_in=3, channels_out=4)
    config.channels_hidden = (4, 5)
    config.up_blocks = [ConcatUp.Config()]
    config.backbone.network.blocks_per_stage = (1, 1)
    config.stem.conv1.conv.stride = stride
    config.stem.conv1.conv.padding = 1
    config.stem.skip = Conv2d.Config(kernel_size=1, stride=stride, padding=0)
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 12, 40, requires_grad=True),
        input_grid=(12, 40),
        batch_size=2,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("stem_stride", [1, 2])
@pytest.mark.parametrize("alignment_stride", [1, 2])
def test_model_cost_keeps_alignment_and_auxiliary_grids_independent(
    stem_stride: int,
    alignment_stride: int,
) -> None:
    config = tiny_model_config(dropout=0)
    config.unet.channels_hidden = (4, 5)
    config.unet.up_blocks = [ConcatUp.Config()]
    config.unet.backbone.network.blocks_per_stage = (1, 1)
    config.unet.stem.conv1.conv.stride = stem_stride
    config.unet.stem.conv1.conv.padding = 0
    config.unet.stem.skip = Conv2d.Config(kernel_size=1, stride=stem_stride, padding=0)
    config.proj_out.stride = alignment_stride
    config.proj_out.padding = 0
    config.reference_columns = 32 // stem_stride // alignment_stride
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 16, 32, requires_grad=True),
        run=_all_cost_outputs,
        input_grid=(16, 32),
        batch_size=2,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("nested", [False, True])
def test_auxiliary_head_cost_retains_nested_running_state(nested: bool) -> None:
    config = tiny_model_config(dropout=0)
    config.unet_gr_rmse_head.elements = [
        Linear.Config(channels_out=4),
        BatchNorm.Config(elementwise_affine=True),
        Activation.Config(),
        Linear.Config(channels_in=4, channels_out=1),
    ]
    if nested:
        config.unet_gr_rmse_head.elements = [
            Sequential.Config(elements=config.unet_gr_rmse_head.elements),
        ]
    execution = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 16, 32, requires_grad=True),
        run=_all_cost_outputs,
        input_grid=(16, 32),
        batch_size=2,
        dtype=torch.float32,
    )
    assert execution.bytes_state == sum(
        value.numel() * value.element_size() for value in config.make().buffers()
    )


def _all_cost_outputs(module: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    if isinstance(module, PosteriorUNet):
        return module(value)
    assert isinstance(module, SequenceModel)
    prediction, extra = module(value)
    return (
        prediction.sum()
        + extra["alignment_logits"].sum()
        + extra["unet_GR_RMSE_pred"].sum()
    )


def test_legacy_alignment_head_keys_load_through_the_output_projection() -> None:
    model = tiny_model_config(dropout=0).make()
    state = model.state_dict()
    legacy = {
        name.replace("proj_out.", "alignment_head."): value
        for name, value in state.items()
    }
    assert "alignment_head.weight" in legacy
    legacy["alignment_head.weight"] = torch.full_like(state["proj_out.weight"], 0.5)
    model.load_state_dict(legacy, strict=True)
    assert torch.equal(model.proj_out.weight, legacy["alignment_head.weight"])


def test_model_level_decoder_values_reach_every_site_only_when_declared() -> None:
    config = tiny_model_config(dropout=-1)
    config.kernel_size = -1
    resolved = config.finalized()
    blocks = [
        resolved.unet.stem,
        resolved.unet.full_up_block.block,
        resolved.unet.out,
        *(up.block for up in resolved.unet.up_blocks),
    ]
    assert len(blocks) == 6
    for block in blocks:
        assert block.conv1.conv.kernel_size == 3
        assert block.drop.p == 0.0
    config.kernel_size = 1
    config.dropout = 0.25
    resolved = config.finalized()
    blocks = [
        resolved.unet.stem,
        resolved.unet.full_up_block.block,
        resolved.unet.out,
        *(up.block for up in resolved.unet.up_blocks),
    ]
    for block in blocks:
        assert block.conv1.conv.kernel_size == block.conv2.kernel_size == 1
        assert block.drop.p == 0.25
        assert block.conv1.drop.p == 0.25
    head = resolved.unet_gr_rmse_head.elements
    assert isinstance(head, list)
    assert isinstance(head[3], Dropout.Config)
    assert head[3].p == 0.25


def test_unet_requires_one_decoder_block_per_skip_resolution() -> None:
    config = PosteriorUNet.Config(channels_in=3, channels_out=4)
    config.channels_hidden = (4, 5)
    config.backbone.network.blocks_per_stage = (1, 1)
    with pytest.raises(ValueError, match="one decoder block per skip"):
        config.finalized().make()


def test_channel_layout_belongs_to_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    layouts: list[torch.memory_format] = []

    def record(
        self: nn.Module,
        *args: object,
        **kwargs: torch.memory_format,
    ) -> nn.Module:
        del args
        layouts.append(kwargs["memory_format"])
        return self

    monkeypatch.setattr(nn.Module, "to", record)
    for channels_last in (True, False):
        config = tiny_model_config(dropout=0)
        config.channels_last = channels_last
        model = config.make()
        model.format_for(torch.device("cpu"))
        model.format_for(torch.device("cuda"))
    assert layouts == [torch.channels_last]


def test_model_requires_its_reference_geometry_and_target_statistics() -> None:
    config = tiny_model_config(dropout=0)
    config.target_stats = None
    with pytest.raises(
        ValueError,
        match=r"^The model requires its target statistics\.$",
    ):
        config.make()
    config = tiny_model_config(dropout=0)
    config.reference_columns = -1
    with pytest.raises(
        ValueError,
        match=r"^The model requires its reference window geometry\.$",
    ):
        config.make()
    config = tiny_model_config(dropout=0)
    config.reference_window = -1.0
    with pytest.raises(ValueError, match="reference window geometry"):
        config.make()


def test_zero_reference_window_is_a_valid_flat_axis() -> None:
    config = tiny_model_config(dropout=0)
    config.reference_window = 0.0
    assert config.target_stats is not None
    mean, std = config.target_stats
    model = config.make()
    assert model.config.reference_window == 0.0
    assert torch.equal(
        model.typewell_tvt_axis,
        torch.full((config.reference_columns,), -mean / std),
    )


def test_typewell_axis_is_float32_on_cpu_whatever_the_construction_context() -> None:
    config = tiny_model_config(dropout=0)
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with torch.device("meta"):
            model = config.make()
    finally:
        torch.set_default_dtype(default)
    assert model.typewell_tvt_axis.dtype == torch.float32
    assert model.typewell_tvt_axis.device.type == "cpu"
    assert "typewell_tvt_axis" not in model.state_dict()
    assert "typewell_tvt_axis_raw" not in model.state_dict()


def test_axis_buffers_are_registered_non_persistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register = nn.Module.register_buffer
    seen: dict[str, object] = {}

    def record(
        module: nn.Module,
        name: str,
        tensor: torch.Tensor | None,
        persistent: bool = True,
    ) -> None:
        seen[name] = persistent
        register(module, name, tensor, persistent)

    monkeypatch.setattr(nn.Module, "register_buffer", record)
    tiny_model_config(dropout=0).make()
    assert seen["typewell_tvt_axis_raw"] is False
    assert seen["typewell_tvt_axis"] is False


@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_channels_last_storage_applies_only_to_cuda_models_that_ask_for_it(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    channels_last: bool,
) -> None:
    config = tiny_model_config(dropout=0)
    config.channels_last = channels_last
    model = config.make()
    calls: list[object] = []

    def record(*, memory_format: torch.memory_format) -> nn.Module:
        calls.append(memory_format)
        return model.unet

    monkeypatch.setattr(model.unet, "to", record)
    model.format_for(torch.device(device))
    assert calls == (
        [torch.channels_last] if device == "cuda" and channels_last else []
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
