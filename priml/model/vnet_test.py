"""Check the two-dimensional VNet on small synthetic images."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

import pytest
import torch

from priml.model.conv import Conv2d, ConvTranspose2d
from priml.model.custom_types import ChannelsInOut
from priml.model.norm import BatchNorm2d
from priml.model.special import Dropout2d
from priml.model.swiglu import elu
from priml.model.vnet import (
    ConvNorm,
    VNet,
    VNetDown,
    VNetInput,
    VNetOutput,
    VNetUp,
)
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from configgle import Fig


def _config() -> VNet.Config:
    config = VNet.Config(channels_in=2, channels_out=5)
    config.channels_hidden = (4, 6, 8, 10, 12)
    return config


def _image() -> Tensor:
    return (
        torch.arange(3 * 2 * 16 * 32, dtype=torch.float32).reshape(3, 2, 16, 32) / 101
    )


@pytest.mark.parametrize(
    "config",
    [
        ConvNorm.Config(),
        VNetInput.Config(),
        VNetDown.Config(),
        VNetOutput.Config(),
        VNet.Config(),
    ],
)
def test_composites_infer_only_unspecified_widths(config: Fig[object]) -> None:
    for channels_in, channels_out in [(3, -1), (-1, 5), (3, 6)]:
        resolved = config.copy_tree()
        assert isinstance(resolved, ChannelsInOut)
        resolved.channels_in, resolved.channels_out = channels_in, channels_out
        resolved = resolved.finalize()
        assert resolved.channels_in == (
            channels_out if channels_in == -1 else channels_in
        )
        assert resolved.channels_out == (
            channels_in if channels_out == -1 else channels_out
        )


def test_defaults_are_the_canonical_five_width_vnet() -> None:
    config = VNet.Config(channels_in=1, channels_out=2).copy_tree().finalize()
    assert config.channels_hidden == (16, 32, 64, 128, 256)
    assert [stage.num_layers for stage in config.down] == [1, 2, 3, 2]
    assert [stage.num_layers for stage in config.up] == [2, 2, 1, 1]
    assert [s.dropout and s.dropout.p for s in config.down] == [None, None, 0.5, 0.5]
    assert [s.dropout and s.dropout.p for s in config.up] == [0.5, 0.5, None, None]
    assert all(stage.dropout_skip.p == 0.5 for stage in config.up)
    assert config.input.layer.conv.kernel_size == 5
    assert config.down[0].proj.conv.stride == 2
    assert [(s.channels_in, s.channels_out) for s in config.down] == [
        (16, 32),
        (32, 64),
        (64, 128),
        (128, 256),
    ]
    assert [(s.channels_in, s.channels_skip) for s in config.up] == [
        (256, 128),
        (256, 64),
        (128, 32),
        (64, 16),
    ]
    assert (config.output.channels_in, config.output.channels_out) == (32, 2)
    stages = (VNetInput, VNetDown, VNetUp, VNetOutput)
    assert all(stage.Config().activation is elu for stage in stages)


def test_forward_shape_and_every_parameter_receives_a_gradient() -> None:
    model = _config().make()
    out = model(_image())
    assert out.shape == (3, 5, 16, 32)
    out.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_state_dict_layout_is_stable() -> None:
    names = list(_config().make().state_dict())
    assert names[:2] == ["input.layer.conv.weight", "input.layer.conv.bias"]
    assert names[-2:] == ["output.proj.weight", "output.proj.bias"]
    for expected in (
        "down.0.proj.conv.weight",
        "down.0.proj.norm.running_mean",
        "down.3.blocks.1.conv.weight",
        "up.0.proj.weight",
        "up.0.norm.weight",
        "up.3.blocks.0.norm.running_var",
        "output.conv.norm.weight",
    ):
        assert expected in names
    assert "down.0.blocks.1.conv.weight" not in names
    assert "up.3.blocks.1.conv.weight" not in names
    assert not any("dropout" in name for name in names)


def test_every_convolution_and_norm_is_a_priml_leaf() -> None:
    model = _config().make()
    assert all(
        isinstance(module, Conv2d | ConvTranspose2d)
        for module in model.modules()
        if isinstance(module, nn.Conv2d | nn.ConvTranspose2d)
    )
    assert all(
        isinstance(module, BatchNorm2d)
        for module in model.modules()
        if isinstance(module, nn.BatchNorm2d)
    )


def test_training_reset_restores_parameters_and_running_statistics() -> None:
    torch.manual_seed(19)
    model = _config().make()
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    model(_image()).square().mean().backward()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    torch.manual_seed(19)
    model.reset_parameters()
    assert all(
        torch.equal(value, initial[name]) for name, value in model.state_dict().items()
    )


def test_channel_dropout_acts_in_training_and_is_off_in_evaluation() -> None:
    config = _config()
    for stage in (*config.down[2:], *config.up[:2]):
        assert stage.dropout is not None
        stage.dropout.p = 0.9
    for stage in config.up:
        stage.dropout_skip.p = 0.9
    torch.manual_seed(5)
    model = config.make()
    image = _image()
    model.eval()
    assert torch.equal(model(image), model(image))
    model.train()
    torch.manual_seed(1)
    first = model(image)
    torch.manual_seed(2)
    assert not torch.equal(first, model(image))


def test_decoder_dropout_zeroes_whole_channels_of_the_skip_and_the_input() -> None:
    def run(*, dropout: float | None, skip: float) -> Tensor:
        config = VNetUp.Config(
            channels_in=4,
            channels_skip=3,
            num_layers=1,
            dropout=None if dropout is None else Dropout2d.Config(p=dropout),
            dropout_skip=Dropout2d.Config(p=skip),
        )
        torch.manual_seed(3)
        up = config.make()
        return up(torch.ones(2, 4, 3, 5), skip=torch.ones(2, 3, 6, 10))

    open_ = run(dropout=None, skip=0.0)
    assert not torch.equal(open_, run(dropout=None, skip=1.0))
    assert not torch.equal(open_, run(dropout=1.0, skip=0.0))
    assert torch.equal(run(dropout=0.0, skip=0.0), open_)


def test_down_stage_dropout_is_applied_before_refinement() -> None:
    def run(dropout: float | None) -> Tensor:
        config = VNetDown.Config(
            channels_in=2,
            channels_out=3,
            num_layers=1,
            dropout=None if dropout is None else Dropout2d.Config(p=dropout),
        )
        torch.manual_seed(3)
        image = torch.arange(3 * 2 * 6 * 10, dtype=torch.float32).reshape(3, 2, 6, 10)
        return config.make()(image.sin())

    assert torch.equal(run(0.0), run(None))
    assert not torch.equal(run(1.0), run(None))


@pytest.mark.parametrize(
    "component",
    ["conv", "input", "down", "transpose", "output", "network"],
)
def test_costs_count_actual_geometry_and_owned_state(component: str) -> None:
    channels, grid = 3, (6, 10)
    config: object
    if component == "conv":
        config = ConvNorm.Config(channels_in=channels, channels_out=5)
    elif component == "input":
        config = VNetInput.Config(channels_in=channels, channels_out=6)
    elif component == "down":
        config = VNetDown.Config(
            channels_in=channels,
            channels_out=5,
            num_layers=2,
            dropout=Dropout2d.Config(p=0.2),
        )
    elif component == "transpose":
        config = ConvTranspose2d.Config(channels_in=channels, channels_out=5)
    elif component == "output":
        config = VNetOutput.Config(channels_in=channels, channels_out=5)
    else:
        config = VNet.Config(channels_in=channels, channels_out=5)
        config.channels_hidden = (6, 8, 10, 12, 14)
        grid = (16, 32)
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, channels, *grid, requires_grad=True),
        input_grid=grid,
        batch_size=2,
        dtype=torch.float32,
    )


def test_skip_decoder_cost_counts_both_inputs() -> None:
    config = VNetUp.Config(
        channels_in=11,
        channels_skip=7,
        dropout=Dropout2d.Config(p=0.2),
    )
    assert_cost_matches_torch(
        config,
        build_input=lambda: (
            torch.randn(2, 11, 3, 5, requires_grad=True),
            torch.randn(2, 7, 6, 10, requires_grad=True),
        ),
        run=_run_up,
        input_grid=(3, 5),
        batch_size=2,
        dtype=torch.float32,
    )


def _run_up(module: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
    assert isinstance(module, VNetUp)
    assert len(inputs) == 2
    return module.forward(inputs[0], skip=inputs[1])


def test_invalid_input_and_stage_geometry_is_rejected() -> None:
    config = VNet.Config(channels_in=1, channels_out=4)
    config.channels_hidden = (2, 2)
    with pytest.raises(
        ValueError,
        match=r"^VNet requires five widths and four down/up stages\.$",
    ):
        config.make()
    config = VNet.Config(channels_in=1, channels_out=4)
    config.down = config.down[:3]
    with pytest.raises(
        ValueError,
        match=r"^VNet requires five widths and four down/up stages\.$",
    ):
        config.make()
    config = VNet.Config(channels_in=1, channels_out=4)
    config.up = config.up[:3]
    with pytest.raises(
        ValueError,
        match=r"^VNet requires five widths and four down/up stages\.$",
    ):
        config.make()
    with pytest.raises(
        ValueError,
        match=r"^Input width must divide the first residual width\.$",
    ):
        VNetInput.Config(channels_in=3, channels_out=2).make()


def test_input_repeats_the_image_channels_into_the_first_residual() -> None:
    config = VNetInput.Config(channels_in=2, channels_out=6)
    layer = config.make()
    assert layer.repeats == 3
    x = torch.arange(3 * 2 * 4 * 6, dtype=torch.float32).reshape(3, 2, 4, 6)
    expected = elu(layer.layer(x) + x.repeat(1, 3, 1, 1))
    assert torch.equal(layer(x), expected)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
