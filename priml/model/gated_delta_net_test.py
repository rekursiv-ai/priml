"""Tests for gated_delta_net module."""

from __future__ import annotations

import pytest
import torch

from priml.model.gated_delta_net import (
    GatedDeltaNet,
    _torch_chunk_gated_delta_rule,
)


def test_gated_delta_net_rejects_non_multiple_v_heads():
    """``heads_v`` must be an integer multiple of ``heads_k``.

    Regression for MODEL-004: the GQA replication
    ``heads_v // heads_k`` silently truncated for non-multiples.
    """
    with pytest.raises(ValueError, match="multiple"):
        GatedDeltaNet.Config(
            channels_in=64,
            heads_k=4,
            heads_v=6,
            channels_k_head=16,
            channels_v_head=16,
        ).make()


def test_gated_delta_net_forward():
    m = GatedDeltaNet.Config(
        channels_in=64,
        heads_k=2,
        heads_v=4,
        channels_k_head=16,
        channels_v_head=16,
    ).make()
    x = torch.randn(2, 8, 64)
    out = m(x)
    assert out.shape == (2, 8, 64)


def test_gated_delta_net_single_token():
    m = GatedDeltaNet.Config(
        channels_in=32,
        heads_k=2,
        heads_v=2,
        channels_k_head=8,
        channels_v_head=8,
    ).make()
    x = torch.randn(1, 1, 32)
    out = m(x)
    assert out.shape == (1, 1, 32)


def test_torch_chunk_fallback_shapes():
    B, S, H, dk, dv = 1, 16, 2, 8, 8
    q = torch.randn(B, S, H, dk)
    k = torch.randn(B, S, H, dk)
    v = torch.randn(B, S, H, dv)
    g = torch.randn(B, S, H)
    beta = torch.rand(B, S, H)
    out, state = _torch_chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        chunk_size=8,
    )
    assert out.shape == (B, S, H, dv)
    assert state is None


def test_torch_chunk_fallback_with_final_state():
    B, S, H, dk, dv = 1, 8, 2, 8, 8
    q = torch.randn(B, S, H, dk)
    k = torch.randn(B, S, H, dk)
    v = torch.randn(B, S, H, dv)
    g = torch.randn(B, S, H)
    beta = torch.rand(B, S, H)
    out, state = _torch_chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        output_final_state=True,
    )
    assert out.shape == (B, S, H, dv)
    assert state is not None
    assert state.shape == (B, H, dk, dv)


def test_gated_delta_net_arbitrary_leading_dims():
    m = GatedDeltaNet.Config(
        channels_in=64,
        heads_k=2,
        heads_v=4,
        channels_k_head=16,
        channels_v_head=16,
    ).make()
    x = torch.randn(2, 3, 8, 64)
    out = m(x)
    assert out.shape == (2, 3, 8, 64)


def test_gated_delta_net_rejects_every_degenerate_dimension():
    """Each count is checked, not just ``heads_k``.

    A zero elsewhere reached torch and failed as a tensor-shape error naming
    no config field: ``heads_v=0`` built a zero-width projection,
    ``conv_kernel_size=0`` produced negative padding.
    """
    with pytest.raises(ValueError, match="heads_k"):
        _ = GatedDeltaNet.Config(channels_in=64, heads_k=0, heads_v=2).make()
    with pytest.raises(ValueError, match="heads_v"):
        _ = GatedDeltaNet.Config(channels_in=64, heads_k=2, heads_v=0).make()
    with pytest.raises(ValueError, match="channels_in"):
        _ = GatedDeltaNet.Config(channels_in=0, heads_k=2, heads_v=2).make()
    with pytest.raises(ValueError, match="channels_k_head"):
        _ = GatedDeltaNet.Config(
            channels_in=64, heads_k=2, heads_v=2, channels_k_head=0
        ).make()
    with pytest.raises(ValueError, match="channels_v_head"):
        _ = GatedDeltaNet.Config(
            channels_in=64, heads_k=2, heads_v=2, channels_v_head=0
        ).make()
    with pytest.raises(ValueError, match="conv_kernel_size"):
        _ = GatedDeltaNet.Config(
            channels_in=64, heads_k=2, heads_v=2, conv_kernel_size=0
        ).make()


def test_gated_delta_net_never_initializes_a_closed_gate():
    """``A_log`` must stay finite: ``log(0)`` would close a head forever.

    ``uniform_(0, 16)`` is half-open and really does return exactly 0.0
    (measured once in 10M draws); the forward exponentiates ``A_log``, so a
    single ``-inf`` is a head that can never open again.
    """
    model = GatedDeltaNet.Config(channels_in=64, heads_k=2, heads_v=2).make()
    for _ in range(20):
        model.reset_parameters()
        assert bool(torch.isfinite(model.A_log).all())


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
