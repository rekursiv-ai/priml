"""Tests for priml.model.mla."""

from __future__ import annotations

import pytest
import torch

from priml.model.mla import MultiHeadLatentAttention
from priml.model.rope import RoPE


def _tiny(q_lora_rank: int | None = None) -> MultiHeadLatentAttention:
    return MultiHeadLatentAttention.Config(
        channels_in=128,
        heads=4,
        channels_qk_nope_head=16,
        channels_qk_rope_head=8,
        channels_v_head=16,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=32,
        rope=RoPE.Config(channels_head=8, base=50_000),
    ).make()


def test_forward_shape():
    m = _tiny()
    x = torch.randn(2, 6, 128)
    out, cache = m(x)
    assert out.shape == (2, 6, 128)
    # Latent cache: ``k`` stores c_kv (kv_lora_rank=32),
    # ``v`` stores k_pe (qk_rope=8). Single shared "head" axis.
    assert cache.k.shape == (2, 1, 6, 32)
    assert cache.v.shape == (2, 1, 6, 8)


def test_forward_with_q_lora():
    m = _tiny(q_lora_rank=64)
    x = torch.randn(1, 5, 128)
    out, _ = m(x)
    assert out.shape == (1, 5, 128)
    # Sanity: q_proj is disabled, LoRA path is active.
    assert m.q_proj is None
    assert m.q_a_proj is not None
    assert m.q_a_layernorm is not None
    assert m.q_b_proj is not None


def test_prealloc_cache_decode():
    m = _tiny()
    cache = m.alloc_kv_cache(batch=2, max_seq=16)
    # Latent cache shapes: [B, 1, max_seq, feat].
    assert cache.k.shape == (2, 1, 16, 32)  # c_kv, kv_lora_rank=32
    assert cache.v.shape == (2, 1, 16, 8)  # k_pe, qk_rope=8
    prompt = torch.randn(2, 5, 128)
    out, cache = m(prompt, cache=cache)
    assert out.shape == (2, 5, 128)
    assert cache.length == 5
    for _ in range(3):
        step = torch.randn(2, 1, 128)
        out, cache = m(step, cache=cache)
        assert out.shape == (2, 1, 128)
    assert cache.length == 8


def test_decode_equivalent_to_full_reforward():
    """Cached incremental decode must match a from-scratch forward."""
    torch.manual_seed(0)
    m = _tiny()
    m.eval()
    prompt = torch.randn(1, 4, 128)
    steps = [torch.randn(1, 1, 128) for _ in range(3)]

    # Path A: full forward over [prompt + steps].
    full_input = torch.cat([prompt, *steps], dim=1)
    with torch.no_grad():
        full_out, _ = m(full_input)

    # Path B: prefill + 3 decode steps via cache.
    cache = m.alloc_kv_cache(batch=1, max_seq=16)
    with torch.no_grad():
        _, cache = m(prompt, cache=cache)
        decode_outs: list[torch.Tensor] = []
        for step in steps:
            out, cache = m(step, cache=cache)
            decode_outs.append(out)
    cached_tail = torch.cat(decode_outs, dim=1)
    assert torch.allclose(full_out[:, 4:], cached_tail, atol=1e-5, rtol=1e-4)


def test_mla_cached_chunk_is_causal():
    """A multi-token chunk decoded against a non-empty MLA cache stays causal.

    Regression for MODEL-001: the absorb-attention mask was gated on
    ``total_len == seq_len``, so a chunk of S>1 tokens decoded against
    a non-empty cache dropped the causal mask.
    """
    torch.manual_seed(0)
    m = _tiny()
    m.eval()
    x = torch.randn(2, 4, 128)
    with torch.no_grad():
        full, _ = m(x)
        cache = m.alloc_kv_cache(batch=2, max_seq=8)
        _, cache = m(x[:, :2], cache=cache)
        chunk, _ = m(x[:, 2:], cache=cache)
    assert torch.allclose(chunk, full[:, 2:], atol=1e-5, rtol=1e-4), (
        f"max diff: {(chunk - full[:, 2:]).abs().max().item():.3e}"
    )


def test_softmax_scale_override():
    m = MultiHeadLatentAttention.Config(
        channels_in=64,
        heads=2,
        channels_qk_nope_head=8,
        channels_qk_rope_head=8,
        channels_v_head=8,
        kv_lora_rank=16,
        softmax_scale=0.25,
    ).make()
    assert m.softmax_scale == 0.25


def test_rejects_bad_config():
    with pytest.raises(ValueError, match="heads"):
        MultiHeadLatentAttention.Config(channels_in=64, heads=0, kv_lora_rank=16).make()
    with pytest.raises(ValueError, match="kv_lora_rank"):
        MultiHeadLatentAttention.Config(channels_in=64, heads=4, kv_lora_rank=0).make()


def test_mla_arbitrary_leading_dims():
    m = _tiny()
    x = torch.randn(2, 3, 5, 128)
    out, cache = m(x)
    assert out.shape == (2, 3, 5, 128)
    assert cache.length == 5


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
