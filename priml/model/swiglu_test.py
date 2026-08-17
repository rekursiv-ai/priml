"""Tests for ffn module."""

from __future__ import annotations

import pytest
import torch

from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU, SwiGLUReluSquared
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
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
    shallow = SwiGLU.Config(channels_in=256, channels_hidden=1024, depth=-1).make()
    deep = SwiGLU.Config(channels_in=256, channels_hidden=1024, depth=3).make()
    assert deep.up_proj.depth == 3
    # depth=3 scales kaiming by 1/sqrt(4)=0.5 vs unscaled depth=-1.
    ratio = deep.up_proj.weight.std().item() / shallow.up_proj.weight.std().item()
    assert abs(ratio - 0.5) < 0.05, f"ratio={ratio:.3f}"


def test_ffn_reset():
    m = SwiGLU.Config(channels_in=64).make()
    m.reset_parameters()


def test_ffn_forward_drops_extra_args():
    m = SwiGLU.Config(channels_in=64).make()
    x = torch.randn(2, 8, 64)
    assert m(x, "extra", key="val").shape == (2, 8, 64)


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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
