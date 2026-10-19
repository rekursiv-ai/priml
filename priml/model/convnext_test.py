"""Exercise native ConvNeXt composition and checkpoint boundaries."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Final, override

import warnings

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional
from treekle import from_plain, to_plain

import pytest
import torch

from priml.cost import cost
from priml.model.conv import Conv2d
from priml.model.convnext import (
    ConvNeXt,
    ConvNeXtBlock,
    ConvNeXtHead,
    ConvNeXtStage,
    ConvNeXtStem,
)
from priml.model.linear import Linear
from priml.model.norm import BatchNorm, BatchNorm2d, LayerNorm, LayerNorm2d
from priml.model.sequential import Sequential
from priml.model.special import Identity
from priml.model.swiglu import SwiGLU
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import read_tensors


_CWD: Final = Path(__file__).resolve().parent


def _binary64_drop_path(maximum: float, count: int) -> tuple[float, ...]:
    return tuple(maximum * index / max(count - 1.0, 1.0) for index in range(count))


def test_injected_schedule_round_trips_in_the_printed_configuration() -> None:
    config = ConvNeXt.Config(channels_hidden=(3, 4), blocks_per_stage=(2, 3))
    config.drop_path_schedule = _binary64_drop_path
    restored = from_plain(to_plain(config), ConvNeXt.Config, allow_imports=True)
    assert restored.drop_path_schedule is _binary64_drop_path
    assert "_binary64_drop_path" in restored.pformat(hide_default_values=False)
    assert restored.block_counts() == (2, 3)


def test_tiny_default_and_direct_downsampler_fields() -> None:
    config = ConvNeXt.Config()
    assert config.blocks_per_stage == (3, 3, 9, 3)
    assert isinstance(config.stage, ConvNeXtStage.Config)
    names = {entry.name for entry in fields(config.stage)}
    assert not names & {"stride", "norm", "proj_in"}
    assert isinstance(config.stage.downsample, Sequential.Config)
    config.channels_hidden = (3, 4)
    config.blocks_per_stage = (2, 3)
    resolved = config.finalized()
    assert isinstance(resolved.stage, list)
    assert isinstance(resolved.stage[0].downsample, Identity.Config)
    assert isinstance(resolved.stage[1].downsample, Sequential.Config)
    assert resolved.feature_grid((24, 40)) == (3, 5)


@pytest.mark.parametrize(
    ("factory", "output_grid"),
    [
        (ConvNeXtStem.Config, (4, 7)),
        (ConvNeXtBlock.Config, (16, 28)),
        (ConvNeXtStage.Config, (8, 14)),
    ],
)
@pytest.mark.parametrize(("channels_in", "channels_out"), [(-1, 5), (3, -1)])
def test_components_infer_either_missing_width_before_execution(
    factory: type[ConvNeXtStem.Config | ConvNeXtBlock.Config | ConvNeXtStage.Config],
    output_grid: tuple[int, int],
    channels_in: int,
    channels_out: int,
) -> None:
    config = factory()
    config.channels_in = channels_in
    config.channels_out = channels_out
    width = max(channels_in, channels_out)
    finalized = config.finalized()
    assert finalized.channels_in == finalized.channels_out == width
    assert finalized.make()(torch.zeros(2, width, 16, 28)).shape == (
        2,
        width,
        *output_grid,
    )


def test_explicit_stages_keep_first_downsample_and_child_initializers() -> None:
    config = ConvNeXt.Config(channels_hidden=(3, 4), blocks_per_stage=(7, 8))
    config.stage = [
        ConvNeXtStage.Config(
            downsample=Conv2d.Config(
                kernel_size=3,
                stride=2,
                padding=1,
                init_weight=torch.nn.init.ones_,
            ),
            blocks_per_stage=2,
        ),
        ConvNeXtStage.Config(
            downsample=Conv2d.Config(kernel_size=3, stride=3, padding=1),
            block=[ConvNeXtBlock.Config(drop_path=0.375)],
        ),
    ]
    config.drop_path = 0.2
    resolved = config.finalized()
    assert isinstance(resolved.stage, list)
    first = resolved.stage[0]
    assert isinstance(first.downsample, Conv2d.Config)
    assert first.downsample.stride == 2
    assert first.downsample.init_weight is torch.nn.init.ones_
    assert config.block_counts() == (2, 1)
    assert isinstance(first.block, list)
    assert (
        tuple(block.drop_path for block in first.block)
        == tuple(
            float(rate) for rate in torch.linspace(0, 0.2, 3, dtype=torch.float32)
        )[:2]
    )
    assert isinstance(resolved.stage[1].block, list)
    assert resolved.stage[1].block[0].drop_path == 0.375
    assert resolved.feature_grid((48, 80)) == (2, 4)
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 48, 80, requires_grad=True),
        input_grid=(48, 80),
        batch_size=2,
        dtype=torch.float32,
    )


def test_injected_schedule_preserves_binary64_rate_bits_and_explicit_blocks() -> None:
    config = ConvNeXt.Config(channels_hidden=(3, 4), blocks_per_stage=(2, 3))
    config.drop_path = 0.1
    config.drop_path_schedule = _binary64_drop_path
    resolved = config.finalized()
    assert isinstance(resolved.stage, list)
    actual = tuple(
        block.drop_path
        for stage in resolved.stage
        if isinstance(stage.block, list)
        for block in stage.block
    )
    expected = tuple(0.1 * index / max(5 - 1.0, 1.0) for index in range(5))
    assert actual == expected
    assert actual != tuple(
        float(value) for value in torch.linspace(0, 0.1, 5, dtype=torch.float32)
    )
    assert isinstance(config.stage, ConvNeXtStage.Config)
    config.stage.block = [ConvNeXtBlock.Config(drop_path=0.375)]
    resolved = config.finalized()
    assert isinstance(resolved.stage, list)
    assert all(
        block.drop_path == 0.375
        for stage in resolved.stage
        if isinstance(stage.block, list)
        for block in stage.block
    )


def _empty_drop_path(maximum: float, count: int) -> tuple[float, ...]:
    del maximum, count
    return ()


def test_schedule_cardinality_is_checked_before_parameter_allocation() -> None:
    config = ConvNeXt.Config(channels_hidden=(3, 4), blocks_per_stage=(2, 3))
    config.drop_path_schedule = _empty_drop_path
    allocations: list[torch.Tensor] = []
    config.stem.conv.init_weight = allocations.append
    with pytest.raises(ValueError, match="one rate per block"):
        config.make()
    assert allocations == []


def test_template_and_declared_stages_keep_all_initial_state_draws_and_gradients() -> (
    None
):
    config = ConvNeXt.Config(channels_hidden=(3, 4), blocks_per_stage=(2, 3))
    config.drop_path = 0.1
    explicit = config.copy_tree()
    assert isinstance(config.stage, ConvNeXtStage.Config)
    first, second = explicit.stage = [
        config.stage.copy_tree(),
        config.stage.copy_tree(),
    ]
    first.downsample = Identity.Config()
    first.blocks_per_stage, second.blocks_per_stage = config.blocks_per_stage
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        model = config.make()
        after = torch.get_rng_state()
        torch.manual_seed(29)
        reference = explicit.make()
        assert torch.equal(torch.get_rng_state(), after)
        assert list(model.state_dict()) == list(reference.state_dict())
        assert "stages.1.downsample.0.weight" in model.state_dict()
        assert "stages.1.downsample.1.weight" in model.state_dict()
        for name, value in model.state_dict().items():
            assert torch.equal(value, reference.state_dict()[name])
        value = (
            torch.arange(2 * 3 * 16 * 24, dtype=torch.float32).reshape(2, 3, 16, 24)
            / 101
        )
        left, right = value.clone().requires_grad_(), value.clone().requires_grad_()
        torch.manual_seed(31)
        actual = model(left)
        actual.square().sum().backward()
        after = torch.get_rng_state()
        torch.manual_seed(31)
        expected = reference(right)
        expected.square().sum().backward()
        assert torch.equal(torch.get_rng_state(), after)
        assert torch.equal(actual, expected)
        assert left.grad is not None
        assert right.grad is not None
        assert torch.equal(left.grad, right.grad)
        for first_parameter, second_parameter in zip(
            model.parameters(),
            reference.parameters(),
            strict=True,
        ):
            assert first_parameter.grad is not None
            assert second_parameter.grad is not None
            assert torch.equal(first_parameter.grad, second_parameter.grad)
        torch.manual_seed(37)
        model.reset_parameters()
        after = torch.get_rng_state()
        torch.manual_seed(37)
        reference.reset_parameters()
        assert torch.equal(torch.get_rng_state(), after)
        for name, value in model.state_dict().items():
            assert torch.equal(value, reference.state_dict()[name])


@pytest.mark.parametrize("component", ["stem", "block", "stage", "head", "network"])
def test_spatial_costs_match_complete_forward_and_backward(component: str) -> None:
    if component == "stem":
        config = ConvNeXtStem.Config(channels_in=3, channels_out=5)
        channels, grid = 3, (28, 44)
    elif component == "block":
        config = ConvNeXtBlock.Config(channels_in=5, drop_path=0.2)
        channels, grid = 5, (3, 7)
    elif component == "stage":
        config = ConvNeXtStage.Config(channels_in=3, channels_out=5)
        channels, grid = 3, (14, 22)
    elif component == "head":
        config = ConvNeXtHead.Config(channels_in=5)
        channels, grid = 5, (3, 7)
    else:
        config = ConvNeXt.Config()
        config.channels_hidden = (5, 7)
        config.blocks_per_stage = (1, 2)
        channels, grid = 3, (24, 40)
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, channels, *grid, requires_grad=True),
        input_grid=grid,
        batch_size=2,
        dtype=torch.float32,
    )


def test_stage_cost_follows_an_injected_convolution_grid() -> None:
    config = ConvNeXtStage.Config(channels_in=3, channels_out=5)
    config.downsample = Conv2d.Config(
        3,
        channels_out=5,
        kernel_size=3,
        stride=2,
        padding=1,
    )
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 14, 22, requires_grad=True),
        input_grid=(14, 22),
        batch_size=2,
        dtype=torch.float32,
    )


def test_stage_propagates_widths_into_an_injected_downsampler() -> None:
    config = ConvNeXtStage.Config(channels_in=3, channels_out=5)
    config.downsample = Conv2d.Config(kernel_size=2, stride=2, padding=0)
    resolved = config.copy_tree().finalize()
    assert isinstance(resolved.downsample, Conv2d.Config)
    assert resolved.downsample.channels_in == 3
    assert resolved.downsample.channels_out == 5
    stage = config.make()
    assert stage(torch.randn(2, 3, 8, 12)).shape == (2, 5, 4, 6)


@pytest.mark.parametrize("network", [False, True])
def test_encoder_cost_omits_only_the_absent_image_gradient(network: bool) -> None:
    if network:
        config = ConvNeXt.Config(channels_hidden=(5, 7), blocks_per_stage=(1, 2))
    else:
        config = ConvNeXtStem.Config(channels_in=3, channels_out=5)
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 24, 40),
        input_grid=(24, 40),
        batch_size=2,
        dtype=torch.float32,
        input_grad=False,
    )


def test_stage_cost_rejects_a_downsampler_that_removes_spatial_axes() -> None:
    config = ConvNeXtStage.Config(channels_in=3, channels_out=5)
    config.downsample = ConvNeXtHead.Config(channels_in=3)
    with pytest.raises(
        ValueError,
        match=r"^ConvNeXt downsamplers must retain two spatial axes\.$",
    ):
        cost(
            config.copy_tree().finalize(),
            input_grid=(7, 11),
            batch_size=2,
            dtype=torch.float32,
        )


def test_stage_cost_requires_a_declared_spatial_transform() -> None:
    config = ConvNeXtStage.Config(channels_in=3, channels_out=3)
    config.downsample = Linear.Config()
    with pytest.raises(TypeError, match="spatial"):
        cost(config.finalized(), input_grid=(7, 11), batch_size=2, dtype=torch.float32)


def test_stage_output_grid_requires_a_declared_downsampler() -> None:
    config = ConvNeXtStage.Config(channels_in=3, channels_out=3)
    config.downsample = _Undeclared.Config()
    with pytest.raises(
        TypeError,
        match=r"^ConvNeXt requires a spatially declared downsampler\.$",
    ):
        config.finalized().output_grid((7, 11))


def test_unfinalized_stage_cost_is_rejected() -> None:
    # An unfinalized stage still holds its block template, not the block list.
    with pytest.raises(AssertionError):
        ConvNeXtStage.Config().cost(input_grid=(7, 11), batch_size=2, dtype=None)


def test_explicit_stages_chain_their_downsampled_feature_grid() -> None:
    config = ConvNeXt.Config(
        channels_in=3,
        channels_hidden=(5, 7),
        blocks_per_stage=(1, 1),
        stage=[ConvNeXtStage.Config(), ConvNeXtStage.Config()],
    ).finalized()
    stem = config.stem.output_grid((32, 48))
    # Each default stage halves both axes with its stride-2 downsampler.
    assert config.feature_grid((32, 48)) == (stem[0] // 4, stem[1] // 4)


def test_explicit_stages_cannot_be_given_another_count() -> None:
    with pytest.raises(
        ValueError,
        match=r"^ConvNeXt width and block stage counts must match\.$",
    ):
        ConvNeXt.Config(
            channels_hidden=(5, 7),
            blocks_per_stage=(1, 1),
            stage=[ConvNeXtStage.Config()],
        ).finalized()


def test_the_network_rejects_a_config_whose_stages_were_edited_after_finalize() -> None:
    config = ConvNeXt.Config(
        channels_in=3,
        channels_hidden=(5,),
        blocks_per_stage=(1,),
    ).finalized()
    assert isinstance(config.stage, list)
    config.stage.append(config.stage[0].copy_tree())
    with pytest.raises(
        ValueError,
        match=r"^ConvNeXt width and block stage counts must match\.$",
    ):
        ConvNeXt(config)


@pytest.mark.parametrize(
    ("channels_in", "channels_out"),
    [(4, -1), (-1, 4)],
)
def test_stem_fills_one_missing_width_from_the_other(
    channels_in: int,
    channels_out: int,
) -> None:
    config = ConvNeXtStem.Config(
        channels_in=channels_in,
        channels_out=channels_out,
    ).finalized()
    assert (config.channels_in, config.channels_out) == (4, 4)
    assert (config.conv.channels_in, config.conv.channels_out) == (4, 4)


def test_encoder_output_grid_requires_a_declared_head_transform() -> None:
    config = ConvNeXt.Config(channels_hidden=(5,), blocks_per_stage=(1,))
    config.head = _Undeclared.Config()
    with pytest.raises(
        TypeError,
        match=r"^ConvNeXt requires a spatially declared head\.$",
    ):
        config.finalized().output_grid((12, 20))


class _Undeclared(nn.Module):
    """A module whose config declares no spatial geometry at all."""

    class Config(Fig["_Undeclared"]):
        pass

    def reset_parameters(self) -> None:
        pass

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return x


def test_block_composes_existing_primitives_in_channel_last_order() -> None:
    config = ConvNeXtBlock.Config()
    config.channels_in = 4
    config.conv_dw.kernel_size = 3
    config.conv_dw.padding = 1
    block = config.make()
    assert isinstance(block.conv_dw, Conv2d)
    assert isinstance(block.norm, LayerNorm)
    assert isinstance(block.mlp, SwiGLU)
    assert block.gamma is not None
    x = torch.arange(2 * 4 * 5 * 7, dtype=torch.float32).reshape(2, 4, 5, 7) / 101
    with host_agnostic_numerics():
        branch = block.conv_dw(x).permute(0, 2, 3, 1)
        branch = functional.layer_norm(
            branch,
            (4,),
            block.norm.weight,
            block.norm.bias,
            block.norm.eps,
        )
        branch = block.mlp.down_proj(functional.gelu(block.mlp.up_proj(branch)))
        # ConvNeXtBlock.gamma broadcasts only over NCHW's channel axis.
        branch = branch.permute(0, 3, 1, 2) * block.gamma.reshape(1, -1, 1, 1)
        assert torch.equal(block(x), branch + x)


def test_stage_downsamples_before_its_ordered_blocks() -> None:
    config = ConvNeXtStage.Config()
    config.channels_in = 3
    config.channels_out = 4
    config.blocks_per_stage = 2
    stage = config.make()
    x = torch.arange(2 * 3 * 8 * 10, dtype=torch.float32).reshape(2, 3, 8, 10) / 101
    assert len(stage.blocks) == 2
    assert torch.equal(stage(x), stage.blocks(stage.downsample(x)))
    assert stage(x).shape == (2, 4, 4, 5)


def test_encoder_exposes_raw_maps_and_postpool_norm() -> None:
    config = ConvNeXt.Config()
    config.channels_hidden = (2, 3, 4, 5)
    config.blocks_per_stage = (1, 1, 1, 1)
    model = config.make()
    # Four downsampling stages require both spatial extents to remain at least two.
    x = torch.arange(2 * 3 * 64 * 96, dtype=torch.float32).reshape(2, 3, 64, 96) / 101
    fmap = model.forward_features(x)
    assert fmap.shape == (2, 5, 2, 3)
    assert config.copy_tree().finalize().feature_grid((64, 96)) == fmap.shape[-2:]
    assert config.copy_tree().finalize().output_grid((64, 96)) == ()
    assert isinstance(model.head, ConvNeXtHead)
    assert torch.equal(model(x), model.head.norm(fmap.mean((-2, -1))))
    assert len(model.stages) == 4
    assert [len(stage.blocks) for stage in model.stages] == [1, 1, 1, 1]


def test_bad_stage_geometry_fails_before_allocating_parameters() -> None:
    config = ConvNeXt.Config()
    config.channels_hidden = (2, 3)
    with pytest.raises(ValueError, match="stage counts"):
        config.make()


@pytest.mark.parametrize("stage_count", [2, 4])
@pytest.mark.parametrize("explicit_blocks", [False, True])
def test_explicit_stage_cardinality_fails_before_stem_allocation(
    stage_count: int,
    explicit_blocks: bool,
) -> None:
    config = ConvNeXt.Config()
    config.channels_hidden = (4, 5, 6)
    config.blocks_per_stage = (1, 1, 1)
    config.stage = [ConvNeXtStage.Config() for _ in range(stage_count)]
    if explicit_blocks:
        for stage in config.stage:
            stage.block = [ConvNeXtBlock.Config()]
    allocations: list[torch.Tensor] = []
    config.stem.conv.init_weight = allocations.append
    with pytest.raises(ValueError, match="stage counts"):
        config.make()
    assert allocations == []


@pytest.mark.parametrize("counts", [(1, 1, 1, 1), (2, 1, 1, 1)])
def test_explicit_block_geometry_preserves_individual_drop_path_edits(
    counts: tuple[int, ...],
) -> None:
    config = ConvNeXt.Config()
    config.channels_hidden = (2, 3, 4, 5)
    config.blocks_per_stage = (1, 1, 1, 1)
    config.drop_path = 0.9
    config.stage = [
        ConvNeXtStage.Config(
            downsample=Identity.Config()
            if index == 0
            else Sequential.Config(
                elements=[
                    LayerNorm2d.Config(),
                    Conv2d.Config(kernel_size=2, stride=2, padding=0, bias=True),
                ],
            ),
            block=[
                ConvNeXtBlock.Config(drop_path=0.125 * offset)
                for offset in range(count)
            ],
        )
        for index, count in enumerate(counts)
    ]
    if counts[0] == 1:
        config.blocks_per_stage = (3, 3, 27, 3)
    model = config.make()
    assert [len(stage.blocks) for stage in model.stages] == list(counts)
    actual = tuple(block.drop_path for stage in model.stages for block in stage.blocks)
    expected = tuple(0.125 * offset for count in counts for offset in range(count))
    assert actual == expected


def test_nested_downsampling_declares_the_executed_grid_and_cost() -> None:
    config = ConvNeXt.Config()
    config.channels_hidden = (5,)
    config.blocks_per_stage = (1,)
    stage = ConvNeXtStage.Config()
    stage.downsample = Sequential.Config(
        elements=[
            Sequential.Config(
                elements=[Conv2d.Config(kernel_size=2, stride=2, padding=0)],
            ),
        ],
    )
    config.stage = [stage]
    resolved = config.finalized()
    value = torch.randn(2, 3, 24, 40)
    actual = resolved.make().forward_features(value)
    assert resolved.feature_grid((24, 40)) == actual.shape[-2:]
    assert_cost_matches_torch(
        config,
        build_input=lambda: value.clone().requires_grad_(),
        input_grid=(24, 40),
        batch_size=2,
        dtype=torch.float32,
    )


def test_explicit_template_blocks_preserve_the_repeated_probability() -> None:
    config = ConvNeXt.Config()
    config.channels_hidden = (2, 3)
    config.blocks_per_stage = (3, 3)
    config.drop_path = 0.9
    assert isinstance(config.stage, ConvNeXtStage.Config)
    config.stage.block = [ConvNeXtBlock.Config(drop_path=0.375)]
    expected = (0.375, 0.375)
    model = config.make()
    assert (
        tuple(block.drop_path for stage in model.stages for block in stage.blocks)
        == expected
    )


@pytest.mark.parametrize("entry", ["forward", "cost"])
def test_residual_rejects_spatial_broadcasting(entry: str) -> None:
    config = ConvNeXtBlock.Config(channels_in=3)
    config.conv_dw.kernel_size = 2
    config.conv_dw.stride = 2
    config.conv_dw.padding = 0
    if entry == "forward":
        with pytest.raises(ValueError, match=r"residual.*grid"):
            # A singleton update is the invalid broadcasting case under test.
            config.make()(torch.randn(2, 3, 2, 2))
    else:
        with pytest.raises(ValueError, match=r"residual.*grid"):
            cost(
                config.finalized(),
                input_grid=(2, 2),
                batch_size=2,
                dtype=torch.float32,
            )


def test_drop_path_schedule_fixes_float32_independently_of_ambient_dtype() -> None:
    config = ConvNeXt.Config()
    config.blocks_per_stage = (1, 1, 1, 1)
    config.drop_path = 0.1
    expected = tuple(
        float(rate) for rate in torch.linspace(0, 0.1, 4, dtype=torch.float32)
    )
    original = torch.get_default_dtype()
    try:
        for dtype in (torch.float32, torch.float64):
            torch.set_default_dtype(dtype)
            assert (
                config.drop_path_schedule(config.drop_path, sum(config.block_counts()))
                == expected
            )
    finally:
        torch.set_default_dtype(original)


def test_stochastic_depth_keeps_eval_and_full_drop_explicit() -> None:
    config = ConvNeXtBlock.Config()
    config.channels_in = 4
    config.drop_path = 1.0
    block = config.make()
    x = torch.ones(2, 4, 3, 5)
    assert block.gamma is not None
    assert torch.equal(block(x), x)
    block.eval()
    # ConvNeXtBlock.gamma broadcasts only over NCHW's channel axis.
    assert torch.equal(
        block(x),
        block.mlp(
            block.norm(block.conv_dw(x).permute(0, 2, 3, 1)),
        ).permute(0, 3, 1, 2)
        * block.gamma.reshape(1, -1, 1, 1)
        + x,
    )


def test_reset_restores_every_child_and_layer_scale() -> None:
    config = ConvNeXt.Config()
    config.channels_hidden = (2, 3, 4, 5)
    config.blocks_per_stage = (1, 1, 1, 1)
    torch.manual_seed(19)
    model = config.make()
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    torch.manual_seed(19)
    model.reset_parameters()
    assert all(
        torch.equal(value, initial[name]) for name, value in model.state_dict().items()
    )


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_invalid_stochastic_depth_is_rejected(probability: float) -> None:
    config = ConvNeXtBlock.Config()
    config.channels_in = 4
    config.drop_path = probability
    with pytest.raises(
        ValueError,
        match=r"^drop_path must be between zero and one[.]$",
    ):
        config.make()


def test_nan_stochastic_depth_fails_before_branch_allocation() -> None:
    config = ConvNeXtBlock.Config()
    config.channels_in = 4
    config.drop_path = float("nan")
    allocations: list[torch.Tensor] = []
    config.conv_dw.init_weight = allocations.append
    with pytest.raises(ValueError, match="drop_path must be between zero and one"):
        config.make()
    assert allocations == []


def test_unfinalized_stage_is_rejected() -> None:
    with pytest.raises(
        AssertionError,
    ):
        ConvNeXtStage(ConvNeXtStage.Config())


def test_block_without_layer_scale_preserves_both_residual_paths() -> None:
    config = ConvNeXtBlock.Config()
    config.channels_in = 4
    config.layer_scale = None
    config.drop_path = 0.5
    block = config.make()
    assert block.gamma is None
    x = torch.arange(2 * 4 * 3 * 5, dtype=torch.float32).reshape(2, 4, 3, 5) / 101
    torch.manual_seed(19)
    # ConvNeXtBlock stochastic depth broadcasts one Bernoulli draw per example.
    expected = x + block.mlp(block.norm(block.conv_dw(x).permute(0, 2, 3, 1))).permute(
        0,
        3,
        1,
        2,
    ) * torch.empty(2, 1, 1, 1).bernoulli_(0.5).div_(0.5)
    torch.manual_seed(19)
    assert torch.equal(block(x), expected)


def test_encoder_replays_complete_external_checkpoint() -> None:
    expected = read_tensors(_CWD / "testdata" / "convnext.pt")
    config = ConvNeXt.Config()
    config.channels_hidden = (2, 3, 4, 5)
    config.blocks_per_stage = (1, 1, 1, 1)
    assert isinstance(config.stage, ConvNeXtStage.Config)
    assert isinstance(config.stage.block, ConvNeXtBlock.Config)
    config.stage.block.conv_dw.kernel_size = 3
    config.stage.block.conv_dw.padding = 1
    with host_agnostic_numerics():
        model = config.make().eval()
        state = {
            name.removeprefix("initial/"): value
            for name, value in expected.items()
            if name.startswith("initial/")
        }
        model.load_timm_state_dict(state)
        # The source witness retains its RGB and four-stage shape constraints.
        x = (
            torch.arange(2 * 3 * 64 * 96, dtype=torch.float32).reshape(2, 3, 64, 96)
            / 101
        )
        assert torch.equal(model.forward_features(x), expected["features"])
        assert torch.equal(model(x), expected["output"])
        saved = {name: value.clone() for name, value in model.state_dict().items()}
        with pytest.raises(ValueError, match="tensor names"):
            model.load_timm_state_dict({})
        wrong = dict(state)
        wrong["stem.0.weight"] = state["stem.0.weight"].double()
        with pytest.raises(ValueError, match="shape or dtype"):
            model.load_timm_state_dict(wrong)
        wrong = dict(state)
        wrong["stem.conv.weight"] = state["stem.0.weight"]
        with pytest.raises(ValueError, match="collide"):
            model.load_timm_state_dict(wrong)
        assert all(
            torch.equal(value, saved[name])
            for name, value in model.state_dict().items()
        )
        with torch.no_grad():
            assert isinstance(model.head, ConvNeXtHead)
            assert isinstance(model.head.norm, LayerNorm)
            assert model.head.norm.weight is not None
            model.head.norm.weight[0] += torch.finfo(torch.float32).eps
        assert not torch.equal(model(x), expected["output"])


@pytest.mark.parametrize("projections", [1, 2])
def test_composed_stage_downsampler_cost_follows_each_grid(projections: int) -> None:
    config = ConvNeXtStage.Config(channels_in=3, channels_out=5)
    config.blocks_per_stage = 1
    config.downsample = Sequential.Config(
        elements=[
            Conv2d.Config(channels_out=5, kernel_size=2, stride=2, padding=0)
            for _ in range(projections)
        ],
    )
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 12, 20, requires_grad=True),
        input_grid=(12, 20),
        batch_size=2,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("component", ["stem", "block", "head"])
def test_injected_normalization_owns_its_carried_state(component: str) -> None:
    if component == "stem":
        config = ConvNeXtStem.Config(channels_in=3, channels_out=5)
        config.norm = BatchNorm2d.Config(elementwise_affine=True)
        grid = (8, 12)
    elif component == "block":
        config = ConvNeXtBlock.Config(channels_in=3)
        config.norm = BatchNorm.Config(elementwise_affine=True)
        grid = (4, 5)
    else:
        config = ConvNeXtHead.Config(channels_in=3)
        config.norm = BatchNorm.Config(elementwise_affine=True)
        grid = (4, 5)
    execution = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, *grid, requires_grad=True),
        input_grid=grid,
        batch_size=2,
        dtype=torch.float32,
    )
    assert execution.bytes_state == sum(
        value.numel() * value.element_size() for value in config.make().buffers()
    )


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("projection", ["linear", "mlp"])
def test_stage_rejects_a_composition_with_undeclared_spatial_behavior(
    nested: bool,
    projection: str,
) -> None:
    downsample = Sequential.Config(
        elements=[
            Linear.Config(8, 4)
            if projection == "linear"
            else SwiGLU.Config(8, 4, channels_hidden=4),
        ],
    )
    if nested:
        downsample = Sequential.Config(elements=[downsample])
    config = ConvNeXtStage.Config(channels_in=3, channels_out=3)
    config.blocks_per_stage = 0
    config.downsample = downsample
    assert config.make()(torch.randn(2, 3, 6, 8)).shape[-2:] == (6, 4)
    finalized = config.finalized()
    with pytest.raises(TypeError, match="spatial"):
        finalized.output_grid((6, 8))
    with pytest.raises(TypeError, match="spatial"):
        cost(finalized, input_grid=(6, 8), batch_size=2, dtype=torch.float32)


def test_invalid_residual_widths_keep_parent_derived_children_inspectable() -> None:
    config = ConvNeXtBlock.Config(channels_in=4, channels_out=8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config.pformat(hide_default_values=False)
        finalized = config.finalized()
    assert finalized.conv_dw.channels_in == finalized.conv_dw.channels_out == 4
    assert finalized.conv_dw.groups == 4
    assert isinstance(finalized.norm, LayerNorm.Config)
    assert finalized.norm.channels_in == 4
    assert isinstance(finalized.mlp, SwiGLU.Config)
    assert finalized.mlp.channels_in == finalized.mlp.channels_out == 4
    with pytest.raises(ValueError, match="channels_in=4 must equal channels_out=8"):
        config.make()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
