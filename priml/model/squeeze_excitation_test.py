"""Compare both projection layouts with their complete native operations."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from torch import Tensor
from torch.nn import functional

import pytest
import torch

from priml.cost import cost, map_cost, set_cost
from priml.model.conv import Conv2d
from priml.model.linear import Linear
from priml.model.squeeze_excitation import SqueezeExcitation
from priml.model.swiglu import relu, sigmoid
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from priml.math.custom_types import TensorFn


def _temporal_squeeze(value: Tensor) -> Tensor:
    return value.mean(dim=-1).view(value.shape[0], value.shape[1])


def _spatial_expand(value: Tensor) -> Tensor:
    # SqueezeExcitation.forward broadcasts each channel gate over both spatial axes.
    return value.view(value.shape[0], value.shape[1], 1, 1)


@pytest.mark.parametrize("convolutional", [False, True])
@pytest.mark.parametrize("gate", [sigmoid, relu])
def test_squeeze_excitation_preserves_both_layouts_and_initialization(
    convolutional: bool,
    gate: TensorFn,
) -> None:
    config = SqueezeExcitation.Config()
    config.channels_in = 3
    config.channels_hidden = 2
    config.gate = gate
    reference_gate = torch.sigmoid if gate is sigmoid else functional.relu
    if convolutional:
        config.fc1 = Conv2d.Config(kernel_size=1, padding=0, bias=True)
        config.fc2 = Conv2d.Config(kernel_size=1, padding=0, bias=True)
        config.act = set_cost(map_cost(primal=5, adjoint=5))(
            partial(functional.silu, inplace=True),
        )
        shape = (2, 3, 4, 5)
    else:
        config.fc1 = Linear.Config(bias=True, init_bias=None)
        config.fc2 = Linear.Config(bias=True, init_bias=None)
        config.squeeze = _temporal_squeeze
        config.expand = _spatial_expand
        # motor_imagery.AttentionEEGNet.conv_s collapses H before this squeeze.
        shape = (2, 3, 1, 5)
    finalized = config.copy_tree().finalize()
    rendered = config.pformat(hide_default_values=False, mask_memory_addresses=True)
    for name in ("channels_hidden", "fc1", "fc2", "squeeze", "expand", "act", "gate"):
        assert f"{name}=" in rendered
    assert isinstance(finalized.fc1, (Linear.Config, Conv2d.Config))
    assert isinstance(finalized.fc2, (Linear.Config, Conv2d.Config))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        native = config.make()
        after = torch.get_rng_state()
        torch.manual_seed(11)
        fc1, fc2 = finalized.fc1.make(), finalized.fc2.make()
        assert torch.equal(torch.get_rng_state(), after)
        for name, value in native.state_dict().items():
            projection, key = name.split(".")
            assert torch.equal(
                value,
                (fc1 if projection == "fc1" else fc2).state_dict()[key],
            )
    assert list(native.state_dict()) == [
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
    ]
    source = (
        torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape) / 31
        - 1
    )
    left, right = source.clone().requires_grad_(), source.clone().requires_grad_()
    if convolutional:
        expected = right * reference_gate(
            fc2(functional.silu(fc1(right.mean((2, 3), keepdim=True)), inplace=True)),
        )
    else:
        # SqueezeExcitation.forward broadcasts each channel gate over both spatial axes.
        expected = right * reference_gate(
            fc2(functional.relu(fc1(right.mean(-1).view(2, 3)))),
        ).view(2, 3, 1, 1)
    actual = native(left, ignored=True)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert left.grad is not None
    assert right.grad is not None
    assert torch.equal(left.grad, right.grad)
    for got, want in zip(
        native.parameters(),
        [*fc1.parameters(), *fc2.parameters()],
        strict=True,
    ):
        assert got.grad is not None
        assert want.grad is not None
        assert torch.equal(got.grad, want.grad)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        native.reset_parameters()
        torch.manual_seed(13)
        fc1.reset_parameters()
        fc2.reset_parameters()
    for got, want in zip(
        native.parameters(),
        [*fc1.parameters(), *fc2.parameters()],
        strict=True,
    ):
        assert torch.equal(got, want)
    assert finalized.output_grid((4, 5)) == (4, 5)


@pytest.mark.parametrize("convolutional", [False, True])
def test_squeeze_excitation_cost_matches_both_projection_layouts(
    convolutional: bool,
) -> None:
    config = SqueezeExcitation.Config()
    config.channels_in = 3
    config.channels_hidden = 2
    if convolutional:
        shape = (2, 3, 4, 5)
    else:
        config.fc1 = Linear.Config()
        config.fc2 = Linear.Config()
        config.squeeze = _temporal_squeeze
        config.expand = _spatial_expand
        # motor_imagery.AttentionEEGNet.conv_s collapses H before this squeeze.
        shape = (2, 3, 1, 5)
    positions = shape[-2] * shape[-1]
    result = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(shape, requires_grad=True),
        input_grid=shape[-2:],
        seq_len=positions,
        batch_size=2,
        dtype=torch.float32,
    )
    assert result.params == 3 * 2 + 2 * 3
    assert result["flops", "adjoint", "reduction"].sum() == 2 * 3 * (positions - 1)
    # Mean spread, ReLU, sigmoid, two product derivatives, and the shared-input add.
    assert result["flops", "adjoint", "elementwise"].sum() == (
        4 * (2 * 3 * positions) + 2 * 2 + 3 * (2 * 3)
    )
    assert result == cost(
        config.copy_tree().finalize(),
        seq_len=positions,
        batch_size=2,
        dtype=torch.float32,
    )


def test_squeeze_excitation_requires_explicit_positive_bottleneck() -> None:
    config = SqueezeExcitation.Config()
    config.channels_in = 3
    with pytest.raises(ValueError, match="channels_hidden"):
        config.make()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
