"""Tests for ffn module."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

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

from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU, SwiGLUReluSquared
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


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
        golden_dir=_TESTDATA,
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
        golden_dir=_TESTDATA,
        golden_name="swiglu_relu_squared",
        build_module=lambda: SwiGLUReluSquared.Config(
            channels_in=4,
            channels_hidden=4,
        ).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_ffn():
    m = SwiGLU.Config(channels_in=64).make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)


def test_ffn_no_gate():
    m = SwiGLU.Config(channels_in=64, gate=False).make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)


def test_ffn_custom_hidden():
    m = SwiGLU.Config(channels_in=64, channels_hidden=128).make()
    # up_proj is fused: 2*128 when gated
    assert m.up_proj.out_features == 256


def test_ffn_channels_infer():
    cfg = SwiGLU.Config(channels_in=64).finalize()
    assert cfg.channels_out == 64

    cfg2 = SwiGLU.Config(channels_out=32).finalize()
    assert cfg2.channels_in == 32


def test_ffn_depth_scales_up_proj_init():
    """``depth`` propagates to projections, scaling init std by 1/sqrt(depth+1).

    Regression for MODEL-002: ``depth`` was stored on the SwiGLU config
    but never forwarded to the projection ``Linear.Config``, so
    depth-scaled init never ran.
    """
    torch.manual_seed(0)
    shallow = SwiGLU.Config(
        channels_in=256, channels_hidden=1024, depth_index=()
    ).make()
    deep = SwiGLU.Config(
        channels_in=256, channels_hidden=1024, depth_index=((3, 4),)
    ).make()
    assert deep.up_proj.depth_index == ((3, 4),)
    # depth_index=((3, 4),) scales kaiming by 1/sqrt(4)=0.5 vs unscaled depth_index=().
    ratio = deep.up_proj.weight.std().item() / shallow.up_proj.weight.std().item()
    assert abs(ratio - 0.5) < 0.05, f"ratio={ratio:.3f}"


def test_ffn_reset():
    m = SwiGLU.Config(channels_in=64).make()
    m.reset_parameters()


def test_ffn_forward_accepts_messages_and_rejects_positional_extras():
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
    ffn = SwiGLUReluSquared.Config(channels_in=8, expansion=3).make()
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


def test_a_norm_is_refused_against_an_ungated_ffn() -> None:
    """The gate norm has no gate to sit inside when ``gate=False``.

    Dropping it silently builds a model the caller did not ask for.
    """
    with pytest.raises(ValueError, match="gate"):
        SwiGLU.Config(channels_in=8, gate=False, norm=RMSNorm.Config()).make()


def test_a_norm_is_refused_against_a_non_silu_activation() -> None:
    """The gate-norm identity ``sigmoid(g) * norm(g * x) == silu(g) * x`` holds
    for silu alone, so pairing it with another activation is a different model
    than the one the norm path was derived for.
    """
    with pytest.raises(ValueError, match="silu"):
        SwiGLU.Config(
            channels_in=8,
            act=torch.nn.functional.gelu,
            norm=RMSNorm.Config(),
        ).make()


def test_gate_norm_width_reset_and_forward_contract() -> None:
    """The hidden width owns the norm, including initialization and arithmetic."""
    config = SwiGLU.Config(
        channels_in=4,
        channels_hidden=6,
        norm=RMSNorm.Config(elementwise_affine=True),
    )
    finalized = config.copy_tree().finalize()
    assert isinstance(finalized.norm, RMSNorm.Config)
    assert finalized.norm.channels_in == 6

    ffn = SwiGLU(config)
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


def test_split_gate_projection_matches_its_separate_biased_matmuls() -> None:
    """Split mode must reuse each half of the fused weight and bias exactly."""
    ffn = SwiGLU.Config(
        channels_in=4,
        channels_hidden=3,
        bias=True,
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
    device_mesh = cast("DeviceMesh", object())
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
