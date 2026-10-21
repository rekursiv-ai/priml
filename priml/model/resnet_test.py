"""Check the post-activation ResNet on small synthetic images."""

from __future__ import annotations

import re

from torch import Tensor, nn

import pytest
import torch

from priml.cost import cost, map_cost, set_cost
from priml.model.resnet import (
    ResNet,
    ResNetBlock,
    ResNetStage,
    relu_inplace,
)
from priml.model.swiglu import relu
from priml.testing.cost import assert_cost_matches_torch


def _config() -> ResNet.Config:
    config = ResNet.Config(channels_in=2, channels_out=7)
    config.stem.channels_out = 3
    config.channels_hidden = (3, 4, 5, 6)
    config.blocks_per_stage = (1, 2, 1, 1)
    return config


def test_defaults_are_the_canonical_resnet34() -> None:
    config = ResNet.Config(channels_in=3, channels_out=1000).copy_tree().finalize()
    assert (config.channels_in, config.channels_out) == (3, 1000)
    assert config.channels_hidden == (64, 128, 256, 512)
    assert config.blocks_per_stage == (3, 4, 6, 3)
    assert (config.stem.channels_in, config.stem.channels_out) == (3, 64)
    assert (config.stem.kernel_size, config.stem.stride) == (7, 2)
    assert (config.proj_out.channels_in, config.proj_out.channels_out) == (512, 1000)
    assert isinstance(config.stage, list)
    assert [stage.stride for stage in config.stage] == [1, 2, 2, 2]
    assert [stage.blocks_per_stage for stage in config.stage] == [3, 4, 6, 3]


def test_forward_shape_and_every_parameter_receives_a_gradient() -> None:
    model = _config().make()
    image = torch.arange(3 * 2 * 32 * 64, dtype=torch.float32).reshape(3, 2, 32, 64)
    logits = model(image / 1_000)
    assert logits.shape == (3, 7)
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_state_dict_layout_is_stable() -> None:
    names = list(_config().make().state_dict())
    assert names[:3] == ["conv1.weight", "bn1.weight", "bn1.bias"]
    assert names[-2:] == ["fc.weight", "fc.bias"]
    assert {re.sub(r"\.\d+(?=\.)", ".N", name) for name in names} >= {
        "stages.N.N.conv1.weight",
        "stages.N.N.norm1.weight",
        "stages.N.N.conv2.weight",
        "stages.N.N.norm2.running_mean",
        "stages.N.N.downsample.N.weight",
        "stages.N.N.downsample.N.running_var",
    }
    assert "stages.1.0.downsample.0.weight" in names
    assert "stages.0.0.downsample.0.weight" not in names
    assert "stages.1.1.downsample.0.weight" not in names


def _torchvision_names(state: dict[str, Tensor]) -> dict[str, Tensor]:
    renamed: dict[str, Tensor] = {}
    for name, value in state.items():
        renamed_name = name
        if name.startswith("stages."):
            renamed_name = re.sub(
                r"^stages\.(\d)\.",
                lambda match: f"layer{int(match[1]) + 1}.",
                name,
            )
            renamed_name = renamed_name.replace(".norm1.", ".bn1.").replace(
                ".norm2.",
                ".bn2.",
            )
        renamed[renamed_name] = value
    return renamed


def test_torchvision_layout_loads_into_the_native_names() -> None:
    source = _config().make()
    for parameter in source.state_dict().values():
        parameter.add_(torch.rand_like(parameter.float()).to(parameter.dtype))
    target = _config().make()
    target.load_torchvision_state_dict(_torchvision_names(source.state_dict()))
    assert all(
        torch.equal(value, source.state_dict()[name])
        for name, value in target.state_dict().items()
    )
    other = _config().make()
    other.load_torchvision_state_dict(source.state_dict())
    assert torch.equal(other.fc.weight, source.fc.weight)


def test_torchvision_load_is_atomic_and_rejects_every_mismatch() -> None:
    model = _config().make()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    torchvision = _torchvision_names(state)
    for bad in (
        dict[str, torch.Tensor](),
        {**torchvision, "fc.bias": torch.zeros(2)},
        {**torchvision, "fc.weight": state["fc.weight"].double()},
        {**torchvision, "stages.0.0.conv1.weight": state["stages.0.0.conv1.weight"]},
    ):
        with pytest.raises(ValueError, match="Checkpoint"):
            model.load_torchvision_state_dict(bad)
        assert all(
            torch.equal(value, state[name])
            for name, value in model.state_dict().items()
        )


def _scales(model: ResNet) -> list[Tensor]:
    return [
        value
        for name, value in model.state_dict().items()
        if name.endswith(".norm2.weight")
    ]


@pytest.mark.parametrize("zero", [False, True])
def test_residual_scale_initializer_is_applied_on_build_and_reset(zero: bool) -> None:
    config = _config()
    assert isinstance(config.stage, ResNetStage.Config)
    assert isinstance(config.stage.block, ResNetBlock.Config)
    if zero:
        config.stage.block.init_scale_residual = nn.init.zeros_
    model = config.make()
    expected = 0.0 if zero else 1.0
    assert len(_scales(model)) == 5
    assert all(scale.eq(expected).all() for scale in _scales(model))
    with torch.no_grad():
        for scale in _scales(model):
            scale.fill_(5.0)
        model.state_dict()["bn1.running_mean"].fill_(3.0)
    model.reset_parameters()
    assert all(scale.eq(expected).all() for scale in _scales(model))
    assert model.state_dict()["bn1.running_mean"].eq(0).all()


def test_residual_scale_is_skipped_for_a_nonaffine_normalization() -> None:
    config = _config()
    assert isinstance(config.stage, ResNetStage.Config)
    assert isinstance(config.stage.block, ResNetBlock.Config)
    config.stage.block.norm2.elementwise_affine = False
    config.stage.block.init_scale_residual = nn.init.zeros_
    model = config.make()
    model.reset_parameters()
    assert model(torch.zeros(3, 2, 32, 64)).shape == (3, 7)


@pytest.mark.parametrize("known", ["in", "out"])
@pytest.mark.parametrize("kind", [ResNetBlock.Config, ResNetStage.Config])
def test_composites_infer_only_the_missing_width(
    kind: type[ResNetBlock.Config | ResNetStage.Config],
    known: str,
) -> None:
    config = (kind(channels_in=6) if known == "in" else kind(channels_out=6)).finalize()
    assert (config.channels_in, config.channels_out) == (6, 6)
    both = kind(channels_in=3, channels_out=6).finalize()
    assert (both.channels_in, both.channels_out) == (3, 6)


def test_stage_template_strides_only_its_first_block() -> None:
    config = ResNetStage.Config(
        channels_in=2,
        channels_out=3,
        stride=2,
        blocks_per_stage=3,
    ).finalize()
    assert isinstance(config.block, list)
    assert [(b.channels_in, b.channels_out, b.stride) for b in config.block] == [
        (2, 3, 2),
        (3, 3, 1),
        (3, 3, 1),
    ]
    assert config.output_grid((8, 12)) == (4, 6)


def test_explicit_blocks_take_the_stage_widths_without_a_template() -> None:
    config = ResNetStage.Config(
        channels_in=2,
        channels_out=4,
        block=[ResNetBlock.Config(), ResNetBlock.Config(stride=2)],
    ).finalize()
    assert isinstance(config.block, list)
    assert [(b.channels_in, b.channels_out) for b in config.block] == [(2, 4), (4, 4)]
    assert [b.stride for b in config.block] == [1, 2]


def test_stage_and_width_counts_must_match() -> None:
    config = ResNet.Config(stage=[ResNetStage.Config(), ResNetStage.Config()])
    with pytest.raises(ValueError, match="stage counts must match"):
        config.finalize()


def test_explicit_stages_keep_their_own_blocks_and_strides() -> None:
    config = ResNet.Config(
        channels_in=2,
        channels_out=3,
        channels_hidden=(4, 5),
        stage=[
            ResNetStage.Config(stride=1, blocks_per_stage=1),
            ResNetStage.Config(stride=1, blocks_per_stage=3),
        ],
    ).finalize()
    assert isinstance(config.stage, list)
    assert [stage.stride for stage in config.stage] == [1, 1]
    assert [stage.blocks_per_stage for stage in config.stage] == [1, 3]


def test_a_projection_exists_only_where_width_or_stride_changes() -> None:
    same = ResNetBlock.Config(channels_in=3, channels_out=3).finalize()
    assert not same.projects()
    assert ResNetBlock.Config(channels_in=3, channels_out=4).finalize().projects()
    assert (
        ResNetBlock.Config(channels_in=3, channels_out=3, stride=2)
        .finalize()
        .projects()
    )
    assert same.make().downsample is None


@pytest.mark.parametrize("projection", [False, True])
def test_residual_block_cost_counts_the_actual_shortcut(projection: bool) -> None:
    config = ResNetBlock.Config(channels_in=3, channels_out=5 if projection else 3)
    config.stride = 2 if projection else 1
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 12, 14, requires_grad=True),
        input_grid=(12, 14),
        batch_size=2,
        dtype=torch.float32,
    )


def test_full_resnet_cost_counts_every_stage_at_its_own_grid() -> None:
    assert_cost_matches_torch(
        _config(),
        build_input=lambda: torch.randn(3, 2, 32, 64, requires_grad=True),
        input_grid=(32, 64),
        batch_size=3,
        dtype=torch.float32,
    )


@set_cost(map_cost(primal=1, adjoint=3))
def _square(x: Tensor) -> Tensor:
    return x.square()


@pytest.mark.parametrize(
    ("activation", "adjoint"),
    [(relu, 7), (relu_inplace, 7), (_square, 21)],
)
def test_activation_cost_uses_the_injected_function_and_operand_dtype(
    activation: object,
    adjoint: int,
) -> None:
    execution = cost(activation, channels=7, dtype=torch.float64)
    assert execution["flops", "primal", "elementwise", torch.float64] == 7
    assert execution["flops", "adjoint", "elementwise", torch.float64] == adjoint
    assert execution["bytes", "primal", "elementwise", torch.float64] == 112
    assert execution["bytes", "adjoint", "elementwise", torch.float64] == 168
    assert execution[torch.float32].sum() == 0


def test_relu_inplace_rectifies_its_input_storage() -> None:
    x = torch.tensor([-2.0, 0.0, 1.5])
    assert relu_inplace(x).data_ptr() == x.data_ptr()
    assert x.tolist() == [0.0, 0.0, 1.5]


def test_unknown_activation_cost_is_rejected_by_the_selected_function() -> None:
    config = ResNetBlock.Config(channels_in=3, channels_out=5)
    config.activation = torch.sin
    with pytest.raises(TypeError, match="sin has no cost"):
        cost(
            config.copy_tree().finalize(),
            input_grid=(6, 8),
            batch_size=2,
            dtype=torch.float32,
        )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
