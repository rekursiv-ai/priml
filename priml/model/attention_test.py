"""Tests for attention module."""

from __future__ import annotations

import pytest
import torch

from priml.model.attention import (
    MultiStreamAttention,
    OutputGate,
    SdpaFused,
    SdpaNaive,
    SelfAttention,
)
from priml.model.kvcache import (
    KVCache,  # used in preallocated cache test
)
from priml.model.norm import RMSNorm
from priml.model.rope import RoPE
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


def test_self_attention():
    m = SelfAttention.Config(channels_in=64, heads=4, channels_head=16).make()
    x = torch.randn(2, 8, 64)
    out, cache = m(x)
    assert out.shape == (2, 8, 64)
    assert cache.length == 8


def test_self_attention_kv_cache():
    m = SelfAttention.Config(channels_in=64, heads=4, channels_head=16).make()
    cache = KVCache.alloc(batch=2, heads=4, max_seq=32, channels_head=16)
    x = torch.randn(2, 8, 64)
    _, cache = m(x, cache=cache)
    assert cache.length == 8
    x2 = torch.randn(2, 1, 64)
    out2, cache = m(x2, cache=cache)
    assert out2.shape == (2, 1, 64)
    assert cache.length == 9


def test_self_attention_preallocated_cache():
    m = SelfAttention.Config(channels_in=64, heads=4, channels_head=16).make()
    cache = KVCache.alloc(batch=2, heads=4, max_seq=32, channels_head=16)
    assert cache.length == 0
    x = torch.randn(2, 8, 64)
    out, cache = m(x, cache=cache)
    assert out.shape == (2, 8, 64)
    assert cache.length == 8
    # Second step
    x2 = torch.randn(2, 1, 64)
    out2, cache = m(x2, cache=cache)
    assert out2.shape == (2, 1, 64)
    assert cache.length == 9


def test_self_attention_gqa():
    m = SelfAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        num_heads_kv=2,
    ).make()
    x = torch.randn(2, 8, 64)
    out, _ = m(x)
    assert out.shape == (2, 8, 64)


def test_self_attention_causal():
    m = SelfAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        causal=True,
    ).make()
    x = torch.randn(2, 8, 64)
    out, _ = m(x)
    assert out.shape == (2, 8, 64)


def test_self_attention_with_rope():
    m = SelfAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        rope=RoPE.Config(channels_head=16),
    ).make()
    x = torch.randn(2, 8, 64)
    out, _ = m(x)
    assert out.shape == (2, 8, 64)


def test_self_attention_with_rope_and_cache():
    m = SelfAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        rope=RoPE.Config(channels_head=16),
    ).make()
    x = torch.randn(2, 8, 64)
    _, cache = m(x)
    x2 = torch.randn(2, 1, 64)
    out2, _ = m(x2, cache=cache)
    assert out2.shape == (2, 1, 64)


def test_self_attention_with_norm_qk():
    m = SelfAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        norm_qk=RMSNorm.Config(16),
    ).make()
    x = torch.randn(2, 8, 64)
    out, _ = m(x)
    assert out.shape == (2, 8, 64)
    # Default: shared instance between Q and K.
    assert m.norm_q is m.norm_k


def test_self_attention_independent_qk_norms():
    m = SelfAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        norm_qk=RMSNorm.Config(channels_in=16, elementwise_affine=True),
        share_qk_norm=False,
    ).make()
    x = torch.randn(2, 8, 64)
    out, _ = m(x)
    assert out.shape == (2, 8, 64)
    # Independent modules, independent parameters.
    norm_q, norm_k = m.norm_q, m.norm_k
    assert isinstance(norm_q, RMSNorm)
    assert isinstance(norm_k, RMSNorm)
    assert norm_q is not norm_k
    with torch.no_grad():
        norm_q.weight.fill_(0.1)
        norm_k.weight.fill_(0.9)
    assert not torch.equal(norm_q.weight, norm_k.weight)
    # reset_parameters runs without double-reset errors.
    m.reset_parameters()


def test_self_attention_reset():
    m = SelfAttention.Config(channels_in=64, heads=4, channels_head=16).make()
    m.reset_parameters()


def test_self_attention_inner_width_differs_from_residual():
    """``channels_in`` (residual) may differ from ``heads*channels_head``.

    Regression for MODEL-008: Qwen3 sets an explicit ``head_dim`` where
    ``hidden_size != heads * head_dim``. SelfAttention must keep the
    residual width (channels_in) separate from the attention inner
    width (heads * channels_head), with ``proj_out`` mapping inner ->
    residual.
    """
    m = SelfAttention.Config(
        channels_in=1024,
        heads=16,
        channels_head=128,
        causal=True,
    ).make()
    assert m.proj_out.weight.shape == (1024, 16 * 128)
    x = torch.randn(2, 4, 1024)
    out, _ = m(x)
    assert out.shape == (2, 4, 1024)


def test_self_attention_channels_infer():
    cfg = SelfAttention.Config(heads=4, channels_head=16).finalize()
    assert cfg.channels_in == 64
    assert cfg.channels_out == 64


def test_self_attention_arbitrary_batch():
    m = SelfAttention.Config(channels_in=64, heads=4, channels_head=16).make()
    x = torch.randn(3, 2, 8, 64)
    out, _ = m(x)
    assert out.shape == (3, 2, 8, 64)


def test_self_attention_cos_sin_kwarg():
    """Supports passing pre-computed cos_sin (sic convention)."""
    rope = RoPE.Config(channels_head=16).make()
    m = SelfAttention.Config(channels_in=64, heads=4, channels_head=16).make()
    x = torch.randn(2, 8, 64)
    cos, sin = rope(torch.arange(8))
    out, _ = m(x, cos_sin=(cos, sin))
    assert out.shape == (2, 8, 64)


def test_self_attention_cached_chunk_is_causal():
    """A multi-token chunk decoded against a non-empty cache stays causal.

    Regression for MODEL-001: ``is_causal`` was gated on
    ``k.shape[-2] == S``, so a chunk of S>1 tokens decoded against a
    non-empty cache silently dropped the causal mask and let earlier
    chunk tokens attend to later ones.
    """
    torch.manual_seed(0)
    m = SelfAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        causal=True,
    ).make()
    x = torch.randn(2, 4, 64)
    full, _ = m(x)
    cache = KVCache.alloc(batch=2, heads=4, max_seq=8, channels_head=16)
    _, cache = m(x[:, :2], cache=cache)
    chunk, _ = m(x[:, 2:], cache=cache)
    assert torch.allclose(chunk, full[:, 2:], atol=1e-5), (
        f"max diff: {(chunk - full[:, 2:]).abs().max().item():.3e}"
    )


def test_self_attention_cached_chunk_rope_positions():
    """Cached-chunk RoPE positions continue from the cache offset.

    Regression guard accompanying MODEL-001/007: a chunk decoded
    against a cache must use absolute positions starting at the cache
    offset so chunked and full forwards agree under RoPE.
    """
    torch.manual_seed(0)
    m = SelfAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        causal=True,
        rope=RoPE.Config(channels_head=16),
    ).make()
    x = torch.randn(2, 4, 64)
    full, _ = m(x)
    cache = KVCache.alloc(batch=2, heads=4, max_seq=8, channels_head=16)
    _, cache = m(x[:, :2], cache=cache)
    chunk, _ = m(x[:, 2:], cache=cache)
    assert torch.allclose(chunk, full[:, 2:], atol=1e-5), (
        f"max diff: {(chunk - full[:, 2:]).abs().max().item():.3e}"
    )


# -- MultiStreamAttention tests --------------------------------------


def test_multi_stream_2_streams():
    m = MultiStreamAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        num_streams=2,
    ).make()
    x0 = torch.randn(2, 8, 64)
    x1 = torch.randn(2, 12, 64)
    y0, y1 = m([x0, x1])
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)


def test_multi_stream_1_stream():
    m = MultiStreamAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        num_streams=1,
    ).make()
    x = torch.randn(2, 16, 64)
    (y,) = m([x])
    assert y.shape == (2, 16, 64)


def test_multi_stream_gqa():
    m = MultiStreamAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        num_heads_kv=2,
        num_streams=2,
    ).make()
    x0 = torch.randn(2, 8, 64)
    x1 = torch.randn(2, 12, 64)
    y0, y1 = m([x0, x1])
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)


def test_multi_stream_with_rope():
    rope = RoPE.Config(channels_head=16).make()
    m = MultiStreamAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        num_streams=2,
    ).make()
    x0 = torch.randn(2, 8, 64)
    x1 = torch.randn(2, 12, 64)
    cs0 = rope(torch.arange(8))
    y0, y1 = m([x0, x1], cos_sin=[cs0, None])
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)


def test_multi_stream_with_norm_qk():
    m = MultiStreamAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        num_streams=2,
        norm_qk=RMSNorm.Config(16),
    ).make()
    x0 = torch.randn(2, 8, 64)
    x1 = torch.randn(2, 12, 64)
    y0, y1 = m([x0, x1])
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)


def test_multi_stream_reset():
    m = MultiStreamAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        num_streams=2,
    ).make()
    m.reset_parameters()


def test_multi_stream_cache():
    m = MultiStreamAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        num_streams=2,
    ).make()
    caches = [
        KVCache.alloc(batch=2, heads=4, max_seq=32, channels_head=16),
        KVCache.alloc(batch=2, heads=4, max_seq=32, channels_head=16),
    ]
    x0 = torch.randn(2, 8, 64)
    x1 = torch.randn(2, 12, 64)
    (y0, y1), caches = m([x0, x1], cache=caches)
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)
    assert caches[0].length == 8
    assert caches[1].length == 12


def test_multi_stream_no_cache_returns_tuple():
    """Without cache kwarg, returns plain tuple of tensors."""
    m = MultiStreamAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        num_streams=2,
    ).make()
    y0, y1 = m([torch.randn(2, 8, 64), torch.randn(2, 12, 64)])
    assert isinstance(y0, torch.Tensor)
    assert isinstance(y1, torch.Tensor)


def test_multi_stream_causal_requires_single_stream():
    with pytest.raises(ValueError, match="causal=True requires num_streams=1"):
        MultiStreamAttention.Config(
            channels_in=64,
            heads=4,
            channels_head=16,
            num_streams=2,
            causal=True,
        ).make()


def test_multi_stream_kv_heads_validation():
    with pytest.raises(ValueError, match="must be divisible"):
        MultiStreamAttention.Config(
            heads=5,
            channels_head=12,
            num_heads_kv=3,
            num_streams=2,
        ).make()


def test_self_attention_kv_heads_validation():
    with pytest.raises(ValueError, match="must be divisible"):
        SelfAttention.Config(
            heads=5,
            channels_head=12,
            num_heads_kv=3,
        ).make()


def test_multi_stream_internal_rope():
    """Internal RoPE via config (not external cos_sin)."""
    m = MultiStreamAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        num_streams=2,
        rope=[RoPE.Config(channels_head=16), None],
    ).make()
    x0 = torch.randn(2, 8, 64)
    x1 = torch.randn(2, 12, 64)
    y0, y1 = m([x0, x1])
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)


# -- SdpaFused / SdpaNaive kernel tests --------------------------------


def test_sdpa_fused_forward():
    kernel = SdpaFused.Config().make()
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    out = kernel(q, k, v)
    assert out.shape == (2, 4, 8, 16)


def test_sdpa_naive_forward():
    kernel = SdpaNaive.Config().make()
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    out = kernel(q, k, v)
    assert out.shape == (2, 4, 8, 16)


def test_naive_matches_fused_noncausal():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    sdp = SdpaFused()(q, k, v)
    eager = SdpaNaive()(q, k, v)
    assert torch.allclose(sdp, eager, atol=1e-6), (
        f"max diff: {(sdp - eager).abs().max().item():.3e}"
    )


def test_naive_matches_fused_causal():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    sdp = SdpaFused()(q, k, v, is_causal=True)
    eager = SdpaNaive()(q, k, v, is_causal=True)
    assert torch.allclose(sdp, eager, atol=1e-6), (
        f"max diff: {(sdp - eager).abs().max().item():.3e}"
    )


def test_naive_causal_masking():
    """Verify future tokens don't influence past positions."""
    kernel = SdpaNaive()
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    out_full = kernel(q, k, v, is_causal=True)
    # Changing k/v at position 3 shouldn't affect output at position 0
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 3, :] = 999.0
    v2[:, :, 3, :] = 999.0
    out_mod = kernel(q, k2, v2, is_causal=True)
    assert torch.equal(out_full[:, :, 0, :], out_mod[:, :, 0, :])


def test_self_attention_with_naive_kernel():
    m = SelfAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        causal=True,
        attn_kernel=SdpaNaive.Config(),
    ).make()
    x = torch.randn(2, 8, 64)
    out, cache = m(x)
    assert out.shape == (2, 8, 64)
    assert cache.length == 8


def test_self_attention_kernel_injection():
    """SdpaNaive and SdpaFused produce numerically close results."""
    torch.manual_seed(0)
    cfg_fused = SelfAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        causal=True,
    )
    cfg_naive = SelfAttention.Config(
        channels_in=64,
        heads=4,
        channels_head=16,
        causal=True,
        attn_kernel=SdpaNaive.Config(),
    )
    m_fused = cfg_fused.make()
    m_naive = cfg_naive.make()
    m_naive.load_state_dict(m_fused.state_dict())
    x = torch.randn(2, 8, 64)
    out_fused, _ = m_fused(x)
    out_naive, _ = m_naive(x)
    assert torch.allclose(out_fused, out_naive, atol=1e-5)


# -- OutputGate tests --------------------------------------------------


def test_output_gate_basic():
    m = OutputGate.Config(
        channels_in=64,
        inner=SelfAttention.Config(
            channels_in=64,
            heads=4,
            channels_head=16,
            causal=True,
        ),
    ).make()
    x = torch.randn(2, 8, 64)
    out, cache = m(x)
    assert out.shape == (2, 8, 64)
    assert cache.length == 8


def test_output_gate_passthrough_kwargs():
    rope = RoPE.Config(channels_head=16).make()
    m = OutputGate.Config(
        channels_in=64,
        inner=SelfAttention.Config(
            channels_in=64,
            heads=4,
            channels_head=16,
        ),
    ).make()
    x = torch.randn(2, 8, 64)
    cos, sin = rope(torch.arange(8))
    out, _ = m(x, cos_sin=(cos, sin))
    assert out.shape == (2, 8, 64)


def test_output_gate_reset():
    m = OutputGate.Config(
        channels_in=64,
        inner=SelfAttention.Config(channels_in=64, heads=4, channels_head=16),
    ).make()
    m.reset_parameters()


def test_output_gate_finalize_propagates():
    cfg = OutputGate.Config(
        channels_in=128,
        inner=SelfAttention.Config(heads=4, channels_head=32),
    ).finalize()
    assert isinstance(cfg.inner, SelfAttention.Config)
    assert cfg.inner.channels_in == 128


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
