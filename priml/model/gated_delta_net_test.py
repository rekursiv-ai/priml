"""Tests for gated_delta_net module."""

from __future__ import annotations

import pytest
import torch

from priml.model.gated_delta_net import (
    GatedDeltaNet,
    _torch_chunk_gated_delta_rule,
)


def test_gated_delta_net_rejects_non_multiple_v_heads():
    """``num_v_heads`` must be an integer multiple of ``num_k_heads``.

    Regression for MODEL-004: the GQA replication
    ``num_v_heads // num_k_heads`` silently truncated for non-multiples.
    """
    with pytest.raises(ValueError, match="multiple"):
        GatedDeltaNet.Config(
            channels_in=64,
            num_k_heads=4,
            num_v_heads=6,
            head_k_dim=16,
            head_v_dim=16,
        ).make()


def test_gated_delta_net_forward():
    m = GatedDeltaNet.Config(
        channels_in=64,
        num_k_heads=2,
        num_v_heads=4,
        head_k_dim=16,
        head_v_dim=16,
    ).make()
    x = torch.randn(2, 8, 64)
    out = m(x)
    assert out.shape == (2, 8, 64)


def test_gated_delta_net_single_token():
    m = GatedDeltaNet.Config(
        channels_in=32,
        num_k_heads=2,
        num_v_heads=2,
        head_k_dim=8,
        head_v_dim=8,
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
        num_k_heads=2,
        num_v_heads=4,
        head_k_dim=16,
        head_v_dim=16,
    ).make()
    x = torch.randn(2, 3, 8, 64)
    out = m(x)
    assert out.shape == (2, 3, 8, 64)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
