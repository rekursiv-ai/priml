"""Tests for ffn module."""

from __future__ import annotations

import torch

from priml.model.swiglu import SwiGLU
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
