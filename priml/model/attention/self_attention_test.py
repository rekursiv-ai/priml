"""Tests for attention module."""

from __future__ import annotations

from pathlib import Path
from typing import override

from configgle import Fig, PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.kvcache import (
    KVCache,  # used in preallocated cache test
)
from priml.model.attention.rope import RoPE
from priml.model.attention.self_attention import SelfAttention
from priml.model.norm import RMSNorm
from priml.testing.bfb import assert_bfb_against_golden, bfb_devices
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)
from priml.testing.golden import assert_text_golden


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


class _LearnedRotary(nn.Module):
    """A custom ``RotaryFactors`` carrying a learned parameter."""

    class Config(Fig["_LearnedRotary"]):
        channels_head: int = 8

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(config.channels_head))

    @override
    def forward(self, positions: Tensor, /) -> tuple[Tensor, Tensor]:
        factors = self.scale.expand(*positions.shape, self.scale.shape[-1])
        return factors.cos(), factors.sin()


def test_self_attention_config_pprint(request: pytest.FixtureRequest) -> None:
    config = SelfAttention.Config(channels_in=16, num_heads=2, channels_head=8)
    assert_text_golden(
        request,
        test_file=__file__,
        name="self_attention",
        rendered=config.pformat(hide_default_values=False),
    )


def test_self_attention_registers_a_custom_rope() -> None:
    """A rotary filled into the slot must join the module tree.

    One left off it is absent from ``state_dict`` and unmoved by
    ``.to(device)``, so a learned variant never trains.
    """
    module = SelfAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        rope=_LearnedRotary.Config(channels_head=16),
    ).make()

    assert "rope" in dict(module.named_modules())
    assert "rope.scale" in dict(module.named_parameters())


def test_self_attention():
    m = SelfAttention.Config(channels_in=64, num_heads=4, channels_head=16).make()
    x = torch.randn(2, 8, 64)
    out = m(x)
    assert out.shape == (2, 8, 64)


def test_self_attention_kv_cache():
    m = SelfAttention.Config(channels_in=64, num_heads=4, channels_head=16).make()
    cache = KVCache.alloc(batch=2, num_heads=4, max_seq=32, channels_head=16)
    x = torch.randn(2, 8, 64)
    _, cache = m.forward_cached(x, cache=cache)
    assert cache.length == 8
    x2 = torch.randn(2, 1, 64)
    out2, cache = m.forward_cached(x2, cache=cache)
    assert out2.shape == (2, 1, 64)
    assert cache.length == 9


def test_self_attention_preallocated_cache():
    m = SelfAttention.Config(channels_in=64, num_heads=4, channels_head=16).make()
    cache = KVCache.alloc(batch=2, num_heads=4, max_seq=32, channels_head=16)
    assert cache.length == 0
    x = torch.randn(2, 8, 64)
    out, cache = m.forward_cached(x, cache=cache)
    assert out.shape == (2, 8, 64)
    assert cache.length == 8
    # Second step
    x2 = torch.randn(2, 1, 64)
    out2, cache = m.forward_cached(x2, cache=cache)
    assert out2.shape == (2, 1, 64)
    assert cache.length == 9


def test_self_attention_gqa():
    m = SelfAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        num_heads_kv=2,
    ).make()
    x = torch.randn(2, 8, 64)
    out = m(x)
    assert out.shape == (2, 8, 64)


def test_self_attention_causal():
    m = SelfAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        causal=True,
    ).make()
    x = torch.randn(2, 8, 64)
    out = m(x)
    assert out.shape == (2, 8, 64)


def test_self_attention_with_rope():
    m = SelfAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        rope=RoPE.Config(channels_head=16),
    ).make()
    x = torch.randn(2, 8, 64)
    out = m(x)
    assert out.shape == (2, 8, 64)


def test_self_attention_with_rope_and_cache():
    m = SelfAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        rope=RoPE.Config(channels_head=16),
    ).make()
    x = torch.randn(2, 8, 64)
    cache = m.alloc_kv_cache(batch=2, max_seq=9)
    _, cache = m.forward_cached(x, cache=cache)
    x2 = torch.randn(2, 1, 64)
    out2, _ = m.forward_cached(x2, cache=cache)
    assert out2.shape == (2, 1, 64)


def test_self_attention_with_norm_qk():
    m = SelfAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        norm_qk=RMSNorm.Config(16),
    ).make()
    x = torch.randn(2, 8, 64)
    out = m(x)
    assert out.shape == (2, 8, 64)
    # Default: shared instance between Q and K.
    assert m.norm_q is m.norm_k


def test_self_attention_norm_qk_channels_inferred_from_channels_head():
    """An unset norm_qk width resolves to channels_head, not channels_in."""
    config = SelfAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        norm_qk=RMSNorm.Config(),
    ).finalize()

    assert isinstance(config.norm_qk, RMSNorm.Config)
    assert config.norm_qk.channels_in == 16
    out = config.make()(torch.randn(2, 8, 64))
    assert out.shape == (2, 8, 64)


def test_self_attention_norm_out_channels_inferred_from_inner_width():
    """An unset norm_out width resolves to num_heads * channels_head.

    An explicit head_dim makes that differ from channels_in (64 vs 128 here),
    so the residual width would be the wrong answer, not merely unresolved.
    """
    config = SelfAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=32,
        norm_out=RMSNorm.Config(),
    ).finalize()

    assert isinstance(config.norm_out, RMSNorm.Config)
    assert config.norm_out.channels_in == 128
    out = config.make()(torch.randn(2, 8, 64))
    assert out.shape == (2, 8, 64)


def test_self_attention_norm_qk_explicit_channels_preserved():
    """An explicit width is the caller's decision; inference must not clobber it."""
    config = SelfAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        norm_qk=RMSNorm.Config(16),
    ).finalize()

    assert isinstance(config.norm_qk, RMSNorm.Config)
    assert config.norm_qk.channels_in == 16


def test_self_attention_independent_qk_norms():
    m = SelfAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        norm_qk=RMSNorm.Config(channels_in=16, elementwise_affine=True),
        share_qk_norm=False,
    ).make()
    x = torch.randn(2, 8, 64)
    out = m(x)
    assert out.shape == (2, 8, 64)
    # Independent modules, independent parameters.
    norm_q, norm_k = m.norm_q, m.norm_k
    assert isinstance(norm_q, RMSNorm)
    assert isinstance(norm_k, RMSNorm)
    assert norm_q is not norm_k
    assert norm_q.weight is not None
    assert norm_k.weight is not None
    with torch.no_grad():
        norm_q.weight.fill_(0.1)
        norm_k.weight.fill_(0.9)
    assert not torch.equal(norm_q.weight, norm_k.weight)
    # reset_parameters runs without double-reset errors.
    m.reset_parameters()


def test_self_attention_reset():
    m = SelfAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        norm_out=RMSNorm.Config(),
    ).make()
    m.reset_parameters()


def test_self_attention_split_qkv_projection() -> None:
    m = SelfAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        bias=True,
        split_qkv_projection=True,
    ).make()

    assert m(torch.randn(2, 4, 16)).shape == (2, 4, 16)


def test_self_attention_head_inference_rejects_ambiguous_or_ragged_dims() -> None:
    with pytest.raises(ValueError, match="Need at least two"):
        SelfAttention.Config().finalize()
    with pytest.raises(ValueError, match="not divisible by num_heads"):
        SelfAttention.Config(channels_in=15, num_heads=2).finalize()
    with pytest.raises(ValueError, match="not divisible by channels_head"):
        SelfAttention.Config(
            channels_in=15,
            num_heads=-1,
            channels_head=8,
        ).finalize()


def test_self_attention_inner_width_differs_from_residual():
    """``channels_in`` (residual) may differ from ``num_heads*channels_head``.

    Regression for MODEL-008: Qwen3 sets an explicit ``head_dim`` where
    ``channels_in != num_heads * head_dim``. SelfAttention must keep the
    residual width (channels_in) separate from the attention inner
    width (num_heads * channels_head), with ``proj_out`` mapping inner ->
    residual.
    """
    m = SelfAttention.Config(
        channels_in=1024,
        num_heads=16,
        channels_head=128,
        causal=True,
    ).make()
    assert m.proj_out.weight.shape == (1024, 16 * 128)
    x = torch.randn(2, 4, 1024)
    out = m(x)
    assert out.shape == (2, 4, 1024)


def test_self_attention_channels_infer():
    cfg = SelfAttention.Config(num_heads=4, channels_head=16).finalize()
    assert cfg.channels_in == 64
    assert cfg.channels_out == 64


def test_self_attention_arbitrary_batch():
    m = SelfAttention.Config(channels_in=64, num_heads=4, channels_head=16).make()
    x = torch.randn(3, 2, 8, 64)
    out = m(x)
    assert out.shape == (3, 2, 8, 64)


def test_self_attention_cos_sin_kwarg():
    """Supports passing pre-computed cos_sin (sic convention)."""
    rope = RoPE.Config(channels_head=16).make()
    m = SelfAttention.Config(channels_in=64, num_heads=4, channels_head=16).make()
    x = torch.randn(2, 8, 64)
    cos, sin = rope(torch.arange(8))
    out = m(x, cos_sin=(cos, sin))
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
        num_heads=4,
        channels_head=16,
        causal=True,
    ).make()
    x = torch.randn(2, 4, 64)
    full = m(x)
    cache = KVCache.alloc(batch=2, num_heads=4, max_seq=8, channels_head=16)
    _, cache = m.forward_cached(x[:, :2], cache=cache)
    chunk, _ = m.forward_cached(x[:, 2:], cache=cache)
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
        num_heads=4,
        channels_head=16,
        causal=True,
        rope=RoPE.Config(channels_head=16),
    ).make()
    x = torch.randn(2, 4, 64)
    full = m(x)
    cache = KVCache.alloc(batch=2, num_heads=4, max_seq=8, channels_head=16)
    _, cache = m.forward_cached(x[:, :2], cache=cache)
    chunk, _ = m.forward_cached(x[:, 2:], cache=cache)
    assert torch.allclose(chunk, full[:, 2:], atol=1e-5), (
        f"max diff: {(chunk - full[:, 2:]).abs().max().item():.3e}"
    )


def test_self_attention_kv_heads_validation():
    with pytest.raises(ValueError, match="must be divisible"):
        SelfAttention.Config(
            num_heads=5,
            channels_head=12,
            num_heads_kv=3,
        ).make()


def test_self_attention_with_naive_kernel():
    m = SelfAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        causal=True,
        attn_kernel=SdpaNaive.Config(),
    ).make()
    x = torch.randn(2, 8, 64)
    out = m(x)
    assert out.shape == (2, 8, 64)


def test_self_attention_forwards_the_open_message_bus() -> None:
    messages: list[object] = []

    def kernel(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        message: object,
        **kwargs: object,
    ) -> Tensor:
        del k, v, kwargs
        messages.append(message)
        return q

    attention = SelfAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        attn_kernel=PartialConfig(kernel),
    ).make()
    message = object()

    attention(torch.randn(1, 4, 16), message=message)

    assert messages == [message]


def test_self_attention_kernel_injection():
    """SdpaNaive and SdpaFused produce numerically close results."""
    torch.manual_seed(0)
    cfg_fused = SelfAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        causal=True,
    )
    cfg_naive = SelfAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        causal=True,
        attn_kernel=SdpaNaive.Config(),
    )
    m_fused = cfg_fused.make()
    m_naive = cfg_naive.make()
    m_naive.load_state_dict(m_fused.state_dict())
    x = torch.randn(2, 8, 64)
    out_fused = m_fused(x)
    out_naive = m_naive(x)
    assert torch.allclose(out_fused, out_naive, atol=1e-5)


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_self_attention_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="self_attention",
        build_module=lambda: (
            SelfAttention.Config(
                channels_in=16,
                num_heads=2,
                channels_head=8,
                causal=True,
            )
            .make()
            .to(device)
        ),
        build_input=lambda: torch.randn(2, 4, 16),
        seed=0,
    )
