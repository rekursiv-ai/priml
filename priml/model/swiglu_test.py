"""Tests for ffn module."""

from __future__ import annotations

from collections.abc import Callable
from inspect import Parameter, signature
from pathlib import Path
from typing import cast, override

from configgle.testing import assert_pprint_golden
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
)

import pytest
import torch

from priml.model.init import kaiming_uniform, unit_fan_in_uniform
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU, SwiGLUReluSquared, relu_squared
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


@pytest.mark.parametrize("config_type", [SwiGLU.Config, SwiGLUReluSquared.Config])
def test_only_channel_boundaries_are_positional(
    config_type: type[SwiGLU.Config],
) -> None:
    config = config_type(4, 8)
    assert (config.channels_in, config.channels_out) == (4, 8)
    positional = [
        name
        for name, parameter in signature(config_type).parameters.items()
        if parameter.kind
        in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional == ["channels_in", "channels_out"]


def test_swiglu_config_pprint() -> None:
    config = SwiGLU.Config(channels_in=4, channels_hidden=4)
    assert_pprint_golden(
        test_file=__file__,
        name="swiglu",
        config=config,
    )


def test_swiglu_relu_squared_config_pprint() -> None:
    config = SwiGLUReluSquared.Config(channels_in=4, channels_hidden=4)
    assert_pprint_golden(
        test_file=__file__,
        name="swiglu_relu_squared",
        config=config,
    )


def test_swiglu_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=Path(__file__).parent.resolve() / "testdata",
        golden_name="swiglu",
        build_module=lambda: SwiGLU.Config(
            channels_in=4,
            channels_hidden=4,
        ).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_swiglu_relu_squared_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=Path(__file__).parent.resolve() / "testdata",
        golden_name="swiglu_relu_squared",
        build_module=lambda: SwiGLUReluSquared.Config(
            channels_in=4,
            channels_hidden=4,
        ).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_ffn() -> None:
    m = SwiGLU.Config(channels_in=64).make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)


def test_ffn_no_gate() -> None:
    m = SwiGLU.Config(channels_in=64, gate=False).make()
    assert m.channels_hidden == 4 * 64
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)


def test_ffn_custom_hidden() -> None:
    m = SwiGLU.Config(channels_in=64, channels_hidden=128).make()
    # up_proj is fused: 2*128 when gated
    assert m.up_proj.out_features == 256


def test_ffn_channels_infer() -> None:
    cfg = SwiGLU.Config(channels_in=64).finalize()
    assert cfg.channels_out == 64

    cfg2 = SwiGLU.Config(channels_out=32).finalize()
    assert cfg2.channels_in == 32


def test_ffn_depth_scales_up_proj_init() -> None:
    """``depth`` propagates to projections, scaling init std by 1/sqrt(depth+1).

    Regression for MODEL-002: ``depth`` was stored on the SwiGLU config
    but never forwarded to the projection ``Linear.Config``, so
    depth-scaled init never ran.
    """
    torch.manual_seed(0)
    shallow = SwiGLU.Config(
        channels_in=256,
        channels_hidden=1024,
        depth_index=(),
        init_weight=kaiming_uniform,
        init_weight_out=kaiming_uniform,
    ).make()
    deep = SwiGLU.Config(
        channels_in=256,
        channels_hidden=1024,
        depth_index=((3, 4),),
        init_weight=kaiming_uniform,
        init_weight_out=kaiming_uniform,
    ).make()
    assert deep.up_proj.depth_index == ((3, 4),)
    # depth_index=((3, 4),) scales kaiming by 1/sqrt(4)=0.5 vs unscaled depth_index=().
    ratio = deep.up_proj.weight.std().item() / shallow.up_proj.weight.std().item()
    assert abs(ratio - 0.5) < 0.05, f"ratio={ratio:.3f}"


def test_ffn_reset() -> None:
    m = SwiGLU.Config(channels_in=64).make()
    m.reset_parameters()


def test_reset_includes_an_empty_norm() -> None:
    ffn = SwiGLU.Config(channels_in=3, channels_hidden=4).make()
    norm = _EmptyResettableNorm()
    assert len(norm) == 0
    ffn.norm = norm
    ffn.reset_parameters()
    assert norm.reset_count == 1


def test_ffn_forward_accepts_messages_and_rejects_positional_extras() -> None:
    m = SwiGLU.Config(channels_in=64).make()
    x = torch.randn(2, 8, 64)
    assert m(x, key="val").shape == (2, 8, 64)
    with pytest.raises(TypeError):
        cast(Callable[..., object], m)(x, "extra")


def test_a_fresh_relu_squared_block_is_the_identity_on_its_residual_stream() -> None:
    """The output projection is zero-initialized, which is the recipe.

    A stack of these starts as shallow as the task needs and deepens as
    training proceeds; a nonzero init would make every layer contribute from
    step one and change what the schedule is tuned against.
    """
    ffn = SwiGLUReluSquared.Config(channels_in=8).make()
    assert torch.equal(ffn(torch.randn(2, 4, 8)), torch.zeros(2, 4, 8))


def test_the_relu_squared_nonlinearity_is_squared() -> None:
    """Squared, not plain: the square is what carries what a gate otherwise
    would, so a plain ReLU is a different model at the same parameter count.
    """
    torch.manual_seed(0)
    ffn = SwiGLUReluSquared.Config(channels_in=8).make()
    with torch.no_grad():
        ffn.down_proj.weight.normal_()
    x = torch.randn(2, 4, 8)
    hidden = torch.relu(ffn.up_proj(x))
    torch.testing.assert_close(ffn.down_proj(hidden.square()), ffn(x), rtol=0, atol=0)


def test_relu_squared_expansion_sets_the_hidden_width() -> None:
    """Ungated, so ``up_proj`` is one matrix wide, not two: the hidden width
    is the only knob, and ``round_to=1`` leaves it an exact multiple.
    """
    ffn = SwiGLUReluSquared.Config(channels_in=8, expansion=3, round_to=1).make()
    assert ffn.up_proj.weight.shape == (24, 8)
    assert ffn.down_proj.weight.shape == (8, 24)


def test_relu_squared_reset_reinitializes_both_projections() -> None:
    """Meta-device materialization drives init through this alone, so a
    projection it skips would train on ``to_empty``'s garbage.
    """
    torch.manual_seed(0)
    ffn = SwiGLUReluSquared.Config(channels_in=8).make()
    with torch.no_grad():
        ffn.up_proj.weight.fill_(float("nan"))
        ffn.down_proj.weight.fill_(float("nan"))
    ffn.reset_parameters()
    assert not torch.isnan(ffn.up_proj.weight).any()
    assert not torch.isnan(ffn.down_proj.weight).any()


def test_ungated_silu_norm_uses_sigmoid_factor() -> None:
    config = SwiGLU.Config()
    config.channels_in = 3
    config.channels_hidden = 4
    config.gate = False
    config.init_weight_out = nn.init.ones_
    config.norm = RMSNorm.Config()
    ffn = config.make()
    assert ffn.act is nn.functional.sigmoid
    assert ffn.norm is not None
    x = torch.tensor([[-2.0, 0.5, 1.0], [1.0, -1.0, 2.0]], requires_grad=True)
    hidden = ffn.up_proj(x)
    expected = ffn.down_proj(torch.sigmoid(hidden) * ffn.norm(hidden))
    actual = ffn(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    inputs = (x, *ffn.parameters())
    expected_grads = torch.autograd.grad(expected.sum(), inputs)
    actual_grads = torch.autograd.grad(actual.sum(), inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("gate", [False, True])
def test_a_norm_is_refused_against_an_unsupported_activation(gate: bool) -> None:
    """An activation without a supported factorization cannot place the norm."""
    with pytest.raises(ValueError, match="silu"):
        SwiGLU.Config(
            channels_in=8,
            gate=gate,
            act=torch.nn.functional.gelu,
            norm=RMSNorm.Config(),
        ).make()


def test_gate_norm_width_reset_and_forward_contract() -> None:
    """The hidden width owns the norm, including initialization and arithmetic."""
    config = SwiGLU.Config(
        channels_in=4,
        channels_hidden=6,
        norm=RMSNorm.Config(elementwise_affine=True),
        init_weight_out=nn.init.ones_,
    )
    finalized = config.copy_tree().finalize()
    assert isinstance(finalized.norm, RMSNorm.Config)
    assert finalized.norm.channels_in == 6

    ffn = config.make()
    assert isinstance(config.norm, RMSNorm.Config)
    assert config.norm.channels_in == -1
    assert ffn.act is nn.functional.sigmoid
    assert isinstance(ffn.norm, RMSNorm)
    assert ffn.norm.normalized_shape == (6,)
    assert ffn.norm.weight is not None
    with torch.no_grad():
        ffn.norm.weight.zero_()
    ffn.reset_parameters()
    assert torch.equal(ffn.norm.weight, torch.ones(6))

    x = torch.randn(2, 3, 4)
    gate, hidden = ffn.up_proj(x).chunk(2, dim=-1)
    expected = ffn.down_proj(torch.sigmoid(gate) * ffn.norm(gate * hidden))
    torch.testing.assert_close(ffn(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("gate", "split"), [(False, False), (True, False), (True, True)]
)
def test_relu_squared_norm_forward_and_gradients(gate: bool, split: bool) -> None:
    config = SwiGLUReluSquared.Config()
    config.channels_in = 3
    config.channels_hidden = 4
    config.gate = gate
    config.split_gate_projection = split
    config.norm = RMSNorm.Config()
    config.norm.elementwise_affine = True
    config.init_weight_out = nn.init.ones_
    ffn = config.make()
    assert ffn.act is nn.functional.relu
    assert isinstance(ffn.norm, RMSNorm)
    assert ffn.norm.normalized_shape == (4,)
    assert ffn.norm.weight is not None
    with torch.no_grad():
        ffn.up_proj.weight.copy_(
            torch.arange(ffn.up_proj.weight.numel()).reshape_as(ffn.up_proj.weight)
            / ffn.up_proj.weight.numel()
            - 0.5
        )
        ffn.norm.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
    x = torch.tensor([[-2.0, 0.5, 1.0], [1.0, -1.0, 2.0]], requires_grad=True)
    projected = ffn.up_proj(x)
    if gate:
        g, hidden = projected.chunk(2, dim=-1)
        norm_input = g * hidden
    else:
        g = norm_input = projected
    expected = ffn.down_proj(torch.relu(g) * ffn.norm(norm_input))
    actual = ffn(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    inputs = (x, *ffn.parameters())
    expected_grads = torch.autograd.grad(expected.sum(), inputs)
    actual_grads = torch.autograd.grad(actual.sum(), inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)
    ffn.reset_parameters()
    assert torch.equal(ffn.norm.weight, torch.ones(4))


def test_split_gate_projection_matches_its_separate_biased_matmuls() -> None:
    """Split mode must reuse each half of the fused weight and bias exactly."""
    ffn = SwiGLU.Config(
        channels_in=4,
        channels_hidden=3,
        bias=True,
        init_weight_out=nn.init.ones_,
        split_gate_projection=True,
    ).make()
    x = torch.randn(2, 3, 4)
    weight = ffn.up_proj.weight
    bias = ffn.up_proj.bias
    assert bias is not None
    gate = torch.matmul(x, weight[:3].T) + bias[:3]
    hidden = torch.matmul(x, weight[3:].T) + bias[3:]
    expected = ffn.down_proj(ffn.act(gate) * hidden)
    torch.testing.assert_close(ffn(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            SwiGLU.Config(channels_in=4, split_gate_projection=True),
            "split_gate_projection",
        ),
        (SwiGLU.Config(channels_in=4, norm=RMSNorm.Config()), "gate norm"),
    ],
)
def test_tensor_parallel_style_refuses_unsupported_gate_paths(
    config: SwiGLU.Config,
    message: str,
) -> None:
    """TP must reject paths whose hidden-axis arithmetic cannot remain aligned."""
    with pytest.raises(NotImplementedError, match=message):
        config.make().tensor_parallel_style()


def test_tensor_parallel_style_preserves_the_logical_gate_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The up output stays sharded as a DTensor until the logical chunk."""
    calls: list[tuple[nn.Module, DeviceMesh, dict[str, ParallelStyle]]] = []

    def fake_parallelize_module(
        module: nn.Module,
        device_mesh: DeviceMesh,
        plan: dict[str, ParallelStyle],
    ) -> nn.Module:
        calls.append((module, device_mesh, plan))
        return module

    monkeypatch.setattr(
        "priml.model.swiglu.parallelize_module",
        fake_parallelize_module,
    )
    ffn = SwiGLU.Config(channels_in=4, channels_hidden=3).make()
    style = ffn.tensor_parallel_style()
    device_mesh = cast(DeviceMesh, object())
    apply_style = cast(
        Callable[[ParallelStyle, nn.Module, DeviceMesh], nn.Module],
        vars(type(style))["_apply"],
    )

    assert apply_style(style, ffn, device_mesh) is ffn
    assert len(calls) == 1
    module, called_mesh, plan = calls[0]
    assert module is ffn
    assert called_mesh is device_mesh
    assert set(plan) == {"up_proj", "down_proj"}
    assert isinstance(plan["up_proj"], ColwiseParallel)
    assert "output_layouts=(Shard(dim=-1),)" in repr(plan["up_proj"])
    assert "use_local_output=False" in repr(plan["up_proj"])
    assert isinstance(plan["down_proj"], RowwiseParallel)
    assert "input_layouts=(Shard(dim=-1),)" in repr(plan["down_proj"])


@pytest.mark.parametrize("recipe", ["base", "custom", "squared"])
def test_legacy_constructor_rng_and_forward_bfb(recipe: str) -> None:
    config = SwiGLUReluSquared.Config() if recipe == "squared" else SwiGLU.Config()
    config.channels_in = 3
    config.bias = True
    config.depth_index = ((2, 3),)
    if recipe == "squared":
        config.round_to = 1
    else:
        config.init_weight = nn.init.normal_ if recipe == "custom" else kaiming_uniform
        config.init_weight_out = config.init_weight
    assert_bfb_against_golden(
        golden_dir=Path(__file__).parent.resolve() / "testdata",
        golden_name=f"swiglu_constructor_{recipe}",
        build_module=nn.Identity,
        build_input=lambda: torch.zeros(1),
        run=lambda _module, _input: _constructor_rng_and_forward(config),
    )


def _constructor_rng_and_forward(config: SwiGLU.Config) -> torch.Tensor:
    # Building inside the runner prevents golden loading from masking init changes.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        module = config.make()
        values = [value.reshape(-1).float() for value in module.state_dict().values()]
        values.append(torch.get_rng_state().float())
        x = torch.randn(2, 3, config.channels_in)
        values.extend([x.reshape(-1), module(x).reshape(-1).float()])
        return torch.cat(values)


@pytest.mark.parametrize("config_type", [SwiGLU.Config, SwiGLUReluSquared.Config])
@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("expansion", [-1, 3.0])
def test_new_default_expansion(
    config_type: type[SwiGLU.Config],
    gate: bool,
    expansion: float,
) -> None:
    config = config_type()
    config.channels_in = 3
    config.gate = gate
    config.expansion = expansion
    config.round_to = 1
    finalized = config.copy_tree().finalize()
    expected = (8 / 3 if gate else 4) if expansion == -1 else expansion
    assert finalized.expansion == expected
    assert finalized.channels_hidden == int(3 * expected)


@pytest.mark.parametrize("config_type", [SwiGLU.Config, SwiGLUReluSquared.Config])
def test_new_default_initializers(config_type: type[SwiGLU.Config]) -> None:
    config = config_type()
    config.channels_in = 3
    config.pprint(hide_default_values=False)
    finalized = config.copy_tree().finalize()
    assert finalized.init_weight is unit_fan_in_uniform
    assert finalized.init_weight_out is nn.init.zeros_


@pytest.mark.parametrize("config_type", [SwiGLU.Config, SwiGLUReluSquared.Config])
def test_new_default_rounded_width(config_type: type[SwiGLU.Config]) -> None:
    config = config_type()
    config.channels_in = 3
    finalized = config.copy_tree().finalize()
    assert finalized.round_to == 256
    assert finalized.channels_hidden == 256
    assert config.make().channels_hidden == 256


@pytest.mark.parametrize("config_type", [SwiGLU.Config, SwiGLUReluSquared.Config])
def test_new_default_projection_init_and_first_gradient(
    config_type: type[SwiGLU.Config],
) -> None:
    config = config_type()
    config.channels_in = 3
    config.channels_hidden = 4
    config.bias = True
    config.depth_index = ((2, 3),)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        module = config.make()
        torch.manual_seed(0)
        expected = torch.empty_like(module.up_proj.weight)
        unit_fan_in_uniform(expected)
        assert torch.equal(module.up_proj.weight, expected)
        assert torch.count_nonzero(module.down_proj.weight) == 0
        x = torch.randn(2, 3, 3, requires_grad=True)
        output = module(x)
        assert torch.equal(output, torch.zeros_like(output))
        output.sum().backward()
        assert x.grad is not None
        assert torch.count_nonzero(x.grad) == 0
        assert module.up_proj.weight.grad is not None
        assert torch.count_nonzero(module.up_proj.weight.grad) == 0
        assert module.down_proj.weight.grad is not None
        assert torch.count_nonzero(module.down_proj.weight.grad) > 0
        with torch.no_grad():
            module.up_proj.weight.fill_(float("nan"))
            module.down_proj.weight.fill_(float("nan"))
        torch.manual_seed(0)
        module.reset_parameters()
        assert torch.equal(module.up_proj.weight, expected)
        assert torch.count_nonzero(module.down_proj.weight) == 0


def test_the_negative_half_is_exactly_zero() -> None:
    """Rectification must preserve the sign distinction before squaring."""
    x = torch.tensor([-3.0, -1e-9, 0.0])
    assert torch.equal(relu_squared(x), torch.zeros(3))


def test_the_positive_half_is_the_square() -> None:
    x = torch.tensor([1e-9, 1.0, 3.0])
    torch.testing.assert_close(relu_squared(x), x.square(), rtol=0, atol=0)


def test_the_gradient_is_a_rectifier() -> None:
    """The squared rectifier's derivative is continuous at the origin."""
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    relu_squared(x).sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, 2 * torch.relu(x.detach()), rtol=0, atol=0)


class _EmptyResettableNorm(nn.Sequential):
    def __init__(self) -> None:
        super().__init__()
        self.reset_count = 0

    def reset_parameters(self) -> None:
        self.reset_count += 1

    @override
    def forward(self, input: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        return super().forward(input)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
