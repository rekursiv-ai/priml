"""Tests for attention module."""

from __future__ import annotations

from pathlib import Path
from typing import Final, override

from configgle import Fig, PartialConfig
from configgle.testing import assert_pprint_golden
from torch import Tensor, nn

import pytest
import torch

from priml.cost import cost, matmul_cost
from priml.model.attention.attention import (
    Attention,
    AttentionProjections,
)
from priml.model.attention.kernel import (
    SdpaFused,
    SdpaNaive,
    attention_kernel_cost,
)
from priml.model.attention.kvcache import (
    KVCache,  # Used in preallocated cache test.
)
from priml.model.attention.multi_stream import MultiStreamAttention
from priml.model.attention.rope import RoPE
from priml.model.attention.window import causal_chunk_mask, window_mask
from priml.model.custom_types import (
    HasForwardCached,
    has_forward_cached,
    is_cached_attention,
)
from priml.model.embedding import Embedding
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.sequential import Sequential
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.transformer import Transformer
from priml.testing.bfb import assert_bfb_against_golden, bfb_devices
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


class _LearnedRotary(nn.Module):
    """A custom ``RotaryFactors`` carrying a learned parameter."""

    class Config(Fig["_LearnedRotary"]):
        channels_head: int = 8

        def rotated_channels(self, channels_head: int) -> int:
            del channels_head
            return self.channels_head

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(config.channels_head))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.scale)

    @override
    def forward(self, positions: Tensor, /) -> tuple[Tensor, Tensor]:
        factors = self.scale.expand(*positions.shape, self.scale.shape[-1])
        return factors.cos(), factors.sin()


def test_self_attention_config_pprint() -> None:
    config = Attention.Config(channels_in=16, num_heads=2, channels_head=8)
    assert_pprint_golden(
        test_file=__file__,
        name="self_attention",
        config=config,
    )


def test_self_attention_registers_a_custom_rope() -> None:
    """A rotary filled into the slot must join the module tree.

    One left off it is absent from ``state_dict`` and unmoved by
    ``.to(device)``, so a learned variant never trains.
    """
    module = Attention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        rope=_LearnedRotary.Config(channels_head=16),
    ).make()

    assert "rope" in dict(module.named_modules())
    assert "rope.scale" in dict(module.named_parameters())


def test_self_attention():
    m = Attention.Config(channels_in=64, num_heads=4, channels_head=16).make()
    x = torch.randn(2, 8, 64)
    out = m(x)
    assert out.shape == (2, 8, 64)


def test_self_attention_kv_cache():
    m = Attention.Config(channels_in=64, num_heads=4, channels_head=16).make()
    cache = KVCache.alloc(batch=2, num_heads=4, max_seq=32, channels_head=16)
    x = torch.randn(2, 8, 64)
    _, cache = m.forward_cached(x, cache=cache)
    assert cache.length == 8
    x2 = torch.randn(2, 3, 64)
    out2, cache = m.forward_cached(x2, cache=cache)
    assert out2.shape == (2, 3, 64)
    assert cache.length == 11


def test_self_attention_preallocated_cache():
    m = Attention.Config(channels_in=64, num_heads=4, channels_head=16).make()
    cache = KVCache.alloc(batch=2, num_heads=4, max_seq=32, channels_head=16)
    assert cache.length == 0
    x = torch.randn(2, 8, 64)
    out, cache = m.forward_cached(x, cache=cache)
    assert out.shape == (2, 8, 64)
    assert cache.length == 8
    # Second step.
    x2 = torch.randn(2, 3, 64)
    out2, cache = m.forward_cached(x2, cache=cache)
    assert out2.shape == (2, 3, 64)
    assert cache.length == 11


def test_self_attention_gqa():
    m = Attention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        num_heads_kv=2,
    ).make()
    x = torch.randn(2, 8, 64)
    out = m(x)
    assert out.shape == (2, 8, 64)


def test_self_attention_causal():
    m = Attention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        causal=True,
    ).make()
    x = torch.randn(2, 8, 64)
    out = m(x)
    assert out.shape == (2, 8, 64)


def test_self_attention_with_rope():
    m = Attention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        rope=RoPE.Config(channels_head=16),
    ).make()
    x = torch.randn(2, 8, 64)
    out = m(x)
    assert out.shape == (2, 8, 64)


def test_self_attention_with_rope_and_cache():
    m = Attention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        rope=RoPE.Config(channels_head=16),
    ).make()
    x = torch.randn(2, 8, 64)
    cache = m.alloc_kv_cache(batch=2, max_seq=9)
    _, cache = m.forward_cached(x, cache=cache)
    x2 = torch.randn(2, 3, 64)
    out2, _ = m.forward_cached(x2, cache=cache)
    assert out2.shape == (2, 3, 64)


def test_self_attention_with_norm_qk():
    m = Attention.Config(
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
    config = Attention.Config(
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
    config = Attention.Config(
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
    config = Attention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        norm_qk=RMSNorm.Config(16),
    ).finalize()

    assert isinstance(config.norm_qk, RMSNorm.Config)
    assert config.norm_qk.channels_in == 16


def test_self_attention_independent_qk_norms():
    m = Attention.Config(
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
    m = Attention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        norm_out=RMSNorm.Config(),
    ).make()
    m.reset_parameters()


def test_self_attention_split_qkv_projection() -> None:
    m = Attention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        bias=True,
        split_qkv_projection=True,
    ).make()

    assert m(torch.randn(2, 4, 16)).shape == (2, 4, 16)


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (Attention.Config(), "Need at least two"),
        (
            Attention.Config(channels_in=15, num_heads=2),
            "not divisible by num_heads",
        ),
        (
            Attention.Config(channels_in=15, num_heads=-1, channels_head=8),
            "not divisible by channels_head",
        ),
    ],
)
def test_self_attention_invalid_head_geometry_prints_before_make_rejects(
    config: Attention.Config,
    match: str,
) -> None:
    rendered = config.pformat(hide_default_values=False)

    assert "Attention.Config" in rendered
    with pytest.raises(ValueError, match=match):
        config.make()


def test_self_attention_inner_width_differs_from_residual():
    """``channels_in`` (residual) may differ from ``num_heads*channels_head``.

    Regression for MODEL-008: Qwen3 sets an explicit ``head_dim`` where
    ``channels_in != num_heads * head_dim``. Attention must keep the
    residual width (channels_in) separate from the attention inner
    width (num_heads * channels_head), with ``proj_out`` mapping inner ->
    residual.
    """
    m = Attention.Config(
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
    cfg = Attention.Config(num_heads=4, channels_head=16).finalize()
    assert cfg.channels_in == 64
    assert cfg.channels_out == 64


def test_self_attention_arbitrary_batch():
    m = Attention.Config(channels_in=64, num_heads=4, channels_head=16).make()
    x = torch.randn(3, 2, 8, 64)
    out = m(x)
    assert out.shape == (3, 2, 8, 64)


def test_self_attention_cos_sin_kwarg():
    """Supports passing pre-computed cos_sin (sic convention)."""
    rope = RoPE.Config(channels_head=16).make()
    m = Attention.Config(channels_in=64, num_heads=4, channels_head=16).make()
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
    m = Attention.Config(
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
    m = Attention.Config(
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


def test_self_attention_explicit_mask_combines_with_implicit_causal():
    """A caller-supplied ``attn_mask`` must combine with ``causal``, not replace it.

    Asserts an all-zero (no-op) mask under ``causal=True`` matches plain
    ``causal=True`` and differs from ``causal=False``.
    """
    torch.manual_seed(0)
    config = Attention.Config(
        channels_in=30,
        num_heads=3,
        channels_head=10,
        causal=True,
        attn_kernel=SdpaNaive.Config(),
    )
    module = config.make()
    module.eval()
    x = torch.randn(2, 5, 30)
    # Attention._forward receives one sequence, so query/key lengths are square.
    permissive_mask = torch.zeros(2, 3, 5, 5)

    with torch.inference_mode():
        out_causal_no_mask = module(x)
        out_causal_with_mask = module(x, attn_mask=permissive_mask)
        module.causal = False
        out_non_causal_with_mask = module(x, attn_mask=permissive_mask)

    assert torch.equal(out_causal_with_mask, out_causal_no_mask)
    assert not torch.equal(out_causal_with_mask, out_non_causal_with_mask)


def test_self_attention_decode_matches_explicit_mask_reference() -> None:
    """Cached decode's default masking matches an explicitly-forced mask.

    Asserts greedy-argmax token ids stay identical across several decode
    steps whether the mask is built by default or forced explicitly.
    """
    torch.manual_seed(0)
    config = Transformer.Config(
        proj_in=Embedding.Config(channels_in=32, shard="vocab"),
        channels_in=16,
        channels_out=32,
        num_layers=2,
        block=TransformerBlock.Config(
            attn=Attention.Config(
                num_heads=4,
                num_heads_kv=2,
                channels_head=4,
                causal=True,
                rope=RoPE.Config(channels_head=4),
            ),
        ),
        proj_out=Sequential.Config(
            elements=[RMSNorm.Config(), Linear.Config(shard="vocab")],
        ),
    )
    model = config.make()
    model.eval()
    prompt = torch.randint(0, 32, (2, 3))

    fixed_tokens = _greedy_decode(model, prompt=prompt, num_steps=8, reference=False)
    reference_tokens = _greedy_decode(model, prompt=prompt, num_steps=8, reference=True)

    assert fixed_tokens == reference_tokens


def test_self_attention_decode_respects_configured_window() -> None:
    """Cached single-token decode must honor a configured attention window.

    Asserts a windowed decode step's output differs from an unwindowed one
    and matches a mask built by hand from ``window_mask``.
    """
    torch.manual_seed(0)
    config = Attention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        causal=True,
        attn_kernel=SdpaFused.Config(),
    )
    module = config.make()
    module.eval()
    x = torch.randn(2, 7, 16)
    window = 2

    def decode(*, window: int) -> Tensor:
        cache = KVCache.alloc(batch=2, num_heads=2, max_seq=8, channels_head=8)
        _, cache = module.forward_cached(x[:, :5], cache=cache, window=window)
        return module.forward_cached(x[:, 5:], cache=cache, window=window)[0]

    with torch.inference_mode():
        windowed = decode(window=window)
        unbounded = decode(window=-1)

    assert not torch.allclose(windowed, unbounded), (
        "a configured window must change the decode step's output; equal "
        "outputs mean `window` silently stopped applying at S == 1 -- the "
        "latent bug this fix corrects"
    )

    with torch.inference_mode():
        cache = KVCache.alloc(batch=2, num_heads=2, max_seq=8, channels_head=8)
        _, cache = module.forward_cached(x[:, :5], cache=cache, window=window)
        forced_mask = window_mask(
            torch.empty(2, 3, 4, dtype=x.dtype),
            torch.empty(cache.seen + 2, 3, 4, dtype=x.dtype),
            window=window,
        )
        forced, _ = module.forward_cached(
            x[:, 5:],
            cache=cache,
            is_causal=False,
            attn_mask=forced_mask,
        )

    assert torch.equal(windowed, forced)


def test_self_attention_cached_chunk_respects_configured_window() -> None:
    """A multi-token cached chunk (S > 1) must honor a configured window too.

    Asserts a windowed chunk's output differs from an unwindowed one and
    matches a mask built by hand from ``window_mask``.
    """
    torch.manual_seed(0)
    config = Attention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        causal=True,
        attn_kernel=SdpaFused.Config(),
    )
    module = config.make()
    module.eval()
    x = torch.randn(2, 8, 16)
    window = 2

    def decode(*, window: int) -> Tensor:
        cache = KVCache.alloc(batch=2, num_heads=2, max_seq=8, channels_head=8)
        _, cache = module.forward_cached(x[:, :5], cache=cache, window=window)
        return module.forward_cached(x[:, 5:], cache=cache, window=window)[0]

    with torch.inference_mode():
        windowed = decode(window=window)
        unbounded = decode(window=-1)

    assert not torch.allclose(windowed, unbounded), (
        "a configured window must change a multi-token cached chunk's "
        "output; equal outputs mean `window` silently stopped applying "
        "at S > 1 -- the gap this fix corrects"
    )

    with torch.inference_mode():
        cache = KVCache.alloc(batch=2, num_heads=2, max_seq=8, channels_head=8)
        _, cache = module.forward_cached(x[:, :5], cache=cache, window=window)
        forced_mask = window_mask(
            torch.empty(3, 2, 4, dtype=x.dtype),
            torch.empty(cache.seen + 3, 2, 4, dtype=x.dtype),
            window=window,
        )
        forced, _ = module.forward_cached(
            x[:, 5:],
            cache=cache,
            is_causal=False,
            attn_mask=forced_mask,
        )

    assert torch.equal(windowed, forced)


def test_self_attention_kv_heads_validation():
    with pytest.raises(ValueError, match="must be divisible"):
        Attention.Config(
            num_heads=5,
            channels_head=12,
            num_heads_kv=3,
        ).make()


def test_self_attention_with_naive_kernel():
    m = Attention.Config(
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

    attention = Attention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        attn_kernel=PartialConfig(kernel),
    ).make()
    message = object()

    attention(torch.randn(2, 4, 16), message=message)

    assert messages == [message]


def test_self_attention_kernel_injection():
    """SdpaNaive and SdpaFused produce numerically close results."""
    torch.manual_seed(0)
    cfg_fused = Attention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        causal=True,
    )
    cfg_naive = Attention.Config(
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
        golden_dir=_CWD / "testdata",
        golden_name="self_attention",
        build_module=lambda: (
            Attention.Config(
                channels_in=4,
                num_heads=2,
                channels_head=2,
                causal=True,
            )
            .make()
            .to(device)
        ),
        build_input=lambda: torch.randn(3, 2, 4),
        seed=0,
    )


@pytest.mark.parametrize("joint", [False, True])
def test_projection_stage_resets_registered_rotary_parameters(joint: bool) -> None:
    stream = AttentionProjections.Config()
    stream.channels_in = 8
    stream.num_heads = 2
    stream.rope = _LearnedRotary.Config()
    stream.rope.channels_head = 2
    if joint:
        cfg = MultiStreamAttention.Config()
        cfg.channels_in = 8
        cfg.num_heads = 2
        cfg.streams = [stream]
        model = cfg.make()
    else:
        model = stream.make()
    model.eval().double()
    rotary = next(m for m in model.modules() if isinstance(m, _LearnedRotary))
    assert not rotary.training
    assert rotary.scale.dtype == torch.float64
    with torch.no_grad():
        rotary.scale.fill_(9)
    model.reset_parameters()
    assert torch.equal(rotary.scale, torch.ones_like(rotary.scale))


def test_projection_stage_width_error_names_the_owner() -> None:
    cfg = AttentionProjections.Config()
    cfg.channels_in = 8
    cfg.channels_out = 16
    cfg.num_heads = 2
    with pytest.raises(ValueError, match="channels_in=8 must equal channels_out=16"):
        cfg.make()


def test_self_attention_cost_is_projections_plus_scores() -> None:
    """Projections follow the matrix rule; scores scale with seq_len, not params."""
    config = Attention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        attn_kernel=SdpaNaive.Config(),
    )
    model_cost = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 32, 16, requires_grad=True),
        seq_len=32,
        batch_size=2,
        dtype=None,
    )
    kernel = attention_kernel_cost(seq_len=32, dtype=None, num_heads=2, channels_head=8)
    qkv = 16 * 8 * (2 + 2 + 2)
    out = 16 * 16
    assert (
        kernel["flops", "primal", "matmul"].sum() == 4 * 2 * 32 * 8 * 32
    )  # QK^T and PV, per head-row.
    assert model_cost.params == qkv + out
    assert (
        model_cost["flops", "primal", "matmul"].sum()
        == 4 * 32 * (qkv + out) + 2 * kernel["flops", "primal", "matmul"].sum()
    )
    assert model_cost["flops", "adjoint", "matmul"].sum() == (
        8 * 32 * (qkv + out) + 2 * kernel["flops", "adjoint", "matmul"].sum()
    )
    assert model_cost.bytes_state == 4 * 2 * 2 * 8


def test_attention_projections_cost_is_its_projections_and_slots() -> None:
    """No kernel here: fused QKV plus the output map, then the injected norms."""
    config = AttentionProjections.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        norm_qk=RMSNorm.Config(elementwise_affine=True),
        share_qk_norm=False,
        norm_out=RMSNorm.Config(elementwise_affine=True),
    )
    finalized = config.copy_tree().finalize()
    model_cost = finalized.cost(seq_len=32, batch_size=1, dtype=None)
    qkv = matmul_cost(channels_in=16, channels_out=8, bias=False, rows=32)
    out = matmul_cost(channels_in=16, channels_out=16, bias=False, rows=32)
    norm_qk = cost(finalized.norm_qk, seq_len=32, batch_size=2, dtype=None)
    norm_out = cost(finalized.norm_out, seq_len=32, batch_size=1, dtype=None)
    assert model_cost["flops", "primal", "matmul"].sum() == (
        6 * qkv["flops", "primal", "matmul"].sum()
        + out["flops", "primal", "matmul"].sum()
    )
    assert (
        model_cost.params
        == 6 * qkv.params + out.params + 2 * norm_qk.params + norm_out.params
    )
    assert model_cost.params == sum(p.numel() for p in config.make().parameters())
    # Two concrete norm invocations (q and k), plus one output invocation.
    assert model_cost["flops", "primal", "elementwise"].sum() == (
        2 * norm_qk["flops", "primal", "elementwise"].sum()
        + norm_out["flops", "primal", "elementwise"].sum()
    )
    assert model_cost.bytes_state == 0


def test_gqa_cost_caches_only_kv_heads() -> None:
    config = Attention.Config(
        channels_in=16,
        num_heads=4,
        channels_head=4,
        num_heads_kv=2,
    )
    cost = config.copy_tree().finalize().cost(seq_len=8, batch_size=1, dtype=None)
    assert cost.bytes_state == 4 * 2 * 2 * 4
    assert cost.params == sum(p.numel() for p in config.make().parameters())


def test_attention_cost_counts_its_norms_and_rotary() -> None:
    config = Attention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        norm_qk=RMSNorm.Config(elementwise_affine=True),
        share_qk_norm=False,
        rope=RoPE.Config(channels_head=8),
        attn_kernel=SdpaNaive.Config(),
    )
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 8, 16, requires_grad=True),
        seq_len=8,
        batch_size=2,
        dtype=None,
    )


def test_attention_projection_counts_weights_once_and_scales_itemsize() -> None:
    config = Attention.Config()
    config.channels_in = 8
    config.num_heads = 2
    config.channels_head = 4
    config.rope = RoPE.Config(4)
    config = config.copy_tree().finalize()
    one = config.cost(seq_len=8, batch_size=1, dtype=torch.bfloat16)
    batch = config.cost(seq_len=8, batch_size=4, dtype=torch.bfloat16)
    assert (
        4 * one["bytes", "primal", "matmul"].sum()
        - batch[
            "bytes",
            "primal",
            "matmul",
        ].sum()
        == 3 * torch.bfloat16.itemsize * one.params
    )
    assert batch.bytes_state == 2 * 2 * 2 * 4
    wide = config.cost(seq_len=8, batch_size=4, dtype=None)
    assert (
        wide["bytes", torch.float32].sum() == batch["bytes", torch.bfloat16].sum() * 2
    )
    assert wide.bytes_state == batch.bytes_state * 2


def test_independent_qk_norms_read_each_owned_scale_once_per_batch() -> None:
    config = AttentionProjections.Config()
    config.channels_in = 8
    config.num_heads = 2
    config.num_heads_kv = 1
    config.norm_qk = RMSNorm.Config()
    config.norm_qk.elementwise_affine = True
    shared = (
        config.copy_tree()
        .finalize()
        .cost(seq_len=4, batch_size=1, dtype=torch.bfloat16)
    )
    config.share_qk_norm = False
    separate = (
        config.copy_tree()
        .finalize()
        .cost(seq_len=4, batch_size=1, dtype=torch.bfloat16)
    )
    assert (
        separate["bytes", "primal", "elementwise"].sum()
        - shared["bytes", "primal", "elementwise"].sum()
        == torch.bfloat16.itemsize * 4
    )


@pytest.mark.parametrize("split", [False, True])
def test_qkv_traffic_reads_input_per_projection_not_per_head(split: bool) -> None:
    config = AttentionProjections.Config()
    config.channels_in = 8
    config.num_heads = 2
    config.num_heads_kv = 1
    config.split_qkv_projection = split
    actual = (
        config.copy_tree()
        .finalize()
        .cost(seq_len=4, batch_size=1, dtype=torch.bfloat16)
    )
    qkv_heads = (2, 1, 1) if split else (2 + 2 * 1,)
    qkv = sum(4 * (8 + heads * 4) + 8 * heads * 4 for heads in qkv_heads)
    output = 4 * (8 + 8) + 8 * 8
    assert actual["bytes", "primal", "matmul"].sum() == 2 * (qkv + output)
    assert actual["bytes", "adjoint", "matmul"].sum() == 4 * (qkv + output)


def test_attention_to_a_memory_of_x_is_self_attention() -> None:
    attention = Attention.Config(channels_in=8, num_heads=2).make()
    x = torch.randn(3, 5, 8)
    torch.testing.assert_close(attention(x, memory=x), attention(x))


def test_attention_to_a_repeated_key_ignores_how_often_it_repeats() -> None:
    attention = Attention.Config(channels_in=8, num_heads=2).make()
    x = torch.randn(3, 5, 8)
    key = torch.randn(8)
    twice = attention(x, memory=key.expand(3, 2, 8))
    seven = attention(x, memory=key.expand(3, 7, 8))
    torch.testing.assert_close(twice, seven)


def test_each_query_reads_only_its_own_rows_memory() -> None:
    attention = Attention.Config(channels_in=8, num_heads=2, num_heads_kv=1).make()
    x = torch.randn(2, 4, 8)
    memory = torch.randn(2, 6, 8)
    out = attention(x, memory=memory)
    changed = memory.clone()
    changed[1] += 1.0
    moved = attention(x, memory=changed)
    torch.testing.assert_close(moved[0], out[0])
    assert not torch.allclose(moved[1], out[1])


@pytest.mark.parametrize("kernel", [SdpaFused.Config(), SdpaNaive.Config()])
def test_a_mask_leaves_out_the_memory_it_excludes(
    kernel: SdpaFused.Config | SdpaNaive.Config,
) -> None:
    attention = Attention.Config(channels_in=8, num_heads=2, attn_kernel=kernel).make()
    x, memory = torch.randn(2, 4, 8), torch.randn(2, 6, 8)
    # Row 0's last two memory positions are padding, excluded additively.
    mask = torch.zeros(2, 6)
    mask[0, 4:] = float("-inf")
    masked = attention(x, memory=memory, attn_mask=mask[:, None, None, :])
    torch.testing.assert_close(masked[0], attention(x[:1], memory=memory[:1, :4])[0])
    torch.testing.assert_close(masked[1], attention(x[1:], memory=memory[1:])[0])


@pytest.mark.parametrize(
    "config",
    [
        Attention.Config(channels_in=8, num_heads=2, causal=True),
        Attention.Config(channels_in=8, num_heads=2, rope=RoPE.Config(channels_head=4)),
    ],
)
def test_attention_to_a_memory_refuses_causality_and_rotary(
    config: Attention.Config,
) -> None:
    with pytest.raises(ValueError, match="memory"):
        config.make()(torch.randn(2, 4, 8), memory=torch.randn(2, 6, 8))


# A block forwards its keyword messages to every attention, so a window or a causal
# flag meant for self-attention reaches a cross-attention too. Each would mask the
# memory silently, or crash when the queries outnumber it.
@pytest.mark.parametrize(("window", "is_causal"), [(1, False), (0, False), (-1, True)])
def test_attention_to_a_memory_refuses_a_window_or_a_causal_call(
    *,
    window: int,
    is_causal: bool,
) -> None:
    attention = Attention.Config(channels_in=8, num_heads=2).make()
    with pytest.raises(ValueError, match="memory"):
        attention(
            torch.randn(2, 4, 8),
            memory=torch.randn(2, 6, 8),
            window=window,
            is_causal=is_causal,
        )


def test_attention_to_a_memory_accepts_an_unbounded_window() -> None:
    attention = Attention.Config(channels_in=8, num_heads=2).make()
    x, memory = torch.randn(2, 4, 8), torch.randn(2, 6, 8)
    torch.testing.assert_close(
        attention(x, memory=memory, window=-1, is_causal=False),
        attention(x, memory=memory),
    )


@pytest.mark.parametrize("bias", [False, True])
def test_projections_read_the_query_heads_and_the_key_value_heads(
    *,
    bias: bool,
) -> None:
    attention = Attention.Config(
        channels_in=7,
        num_heads=6,
        num_heads_kv=3,
        channels_head=8,
        bias=bias,
    ).make()
    weight, shift = attention.proj_qkv.weight, attention.proj_qkv.bias
    if shift is None:
        shift = torch.zeros(12, 8)
    else:
        nn.init.normal_(shift)
    x, memory = torch.randn(2, 4, 7), torch.randn(2, 5, 7)
    on_x = torch.einsum("...c,edc->...ed", x, weight) + shift
    on_memory = torch.einsum("...c,edc->...ed", memory, weight) + shift
    k, v = attention.project_memory(memory)
    torch.testing.assert_close(attention.project_queries(x), on_x[..., :6, :])
    torch.testing.assert_close(k, on_memory[..., 6:9, :])
    torch.testing.assert_close(v, on_memory[..., 9:, :])


def test_memory_attention_attends_projected_queries_to_projected_memory() -> None:
    config = Attention.Config(
        channels_in=7,
        num_heads=6,
        num_heads_kv=3,
        channels_head=8,
    )
    config.norm_qk = RMSNorm.Config(elementwise_affine=True)
    config.share_qk_norm = False
    attention = config.make()
    assert attention.norm_q is not None
    assert attention.norm_k is not None
    x, memory = torch.randn(2, 4, 7), torch.randn(2, 5, 7)
    k, v = attention.project_memory(memory)
    out = nn.functional.scaled_dot_product_attention(
        attention.norm_q(attention.project_queries(x)).transpose(-3, -2),
        attention.norm_k(k).transpose(-3, -2),
        v.transpose(-3, -2),
        enable_gqa=True,
    )
    torch.testing.assert_close(
        attention(x, memory=memory),
        attention.proj_out(out.transpose(-3, -2).flatten(-2)),
    )


@pytest.mark.parametrize("share_qk_norm", [False, True])
def test_cross_attention_cost_matches_torch(*, share_qk_norm: bool) -> None:
    config = Attention.Config(channels_in=8, num_heads=2, num_heads_kv=1, bias=True)
    config.norm_qk = RMSNorm.Config(elementwise_affine=True)
    config.share_qk_norm = share_qk_norm
    # Torch's FLOP counter sees the naive kernel's matmuls, not fused SDPA's.
    config.attn_kernel = SdpaNaive.Config()
    assert_cost_matches_torch(
        config,
        build_input=lambda: (
            torch.randn(2, 3, 8, requires_grad=True),
            torch.randn(2, 5, 8, requires_grad=True),
        ),
        run=_attend_to_memory,
        seq_len=3,
        batch_size=2,
        dtype=None,
        memory_len=5,
    )


def test_cross_attention_cost_carries_no_cache_and_no_rotary() -> None:
    """Attention to a memory refuses a cache and a rotary embedding.

    So its cost holds no per-token state, and a ``rope`` slot adds no work.
    """
    config = Attention.Config(channels_in=24, num_heads=4, num_heads_kv=2)
    plain = (
        config.copy_tree()
        .finalize()
        .cost(seq_len=3, batch_size=5, dtype=None, memory_len=7)
    )
    config.rope = RoPE.Config(channels_head=6)
    rotary = (
        config.copy_tree()
        .finalize()
        .cost(seq_len=3, batch_size=5, dtype=None, memory_len=7)
    )
    assert plain.bytes_state == 0
    assert rotary == plain


# ``reference=True`` forces the same mask the module would build on its own, by
# explicitly passing ``is_causal=False`` and a ``causal_chunk_mask`` built from each
# step's shapes on every decode call -- the mask depends only on shape/dtype/device
# (never tensor values), so a correctly-shaped dummy tensor reproduces exactly what
# ``_forward`` computes from the real q/k. The prefill step is square (S == T), where
# ``causal_chunk_mask`` returns None and raw ``is_causal`` already applies correctly,
# so both runs use the module's default (unforced) resolution there.
def _greedy_decode(
    model: Transformer,
    prompt: Tensor,
    *,
    num_steps: int,
    reference: bool,
) -> list[int]:
    """Greedy-decode ``num_steps`` tokens, optionally forcing the explicit mask."""
    blocks: list[HasForwardCached[object]] = []
    caches: list[object] = []
    for block in model.blocks:
        attn = getattr(block, "attn", None)
        assert is_cached_attention(attn)
        assert has_forward_cached(block)
        caches.append(attn.alloc_kv_cache(batch=prompt.shape[0], max_seq=64))
        blocks.append(block)

    proj_in = model.proj_in
    assert proj_in is not None
    x = proj_in(prompt)
    for i, block in enumerate(blocks):
        x, caches[i] = block.forward_cached(x, cache=caches[i])
    logits = model.project_to_logits(x[:, -1:, :])

    tokens: list[int] = []
    with torch.inference_mode():
        for _ in range(num_steps):
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.extend(int(token) for token in next_token.flatten())
            x = proj_in(next_token)
            for i, block in enumerate(blocks):
                extra: dict[str, object] = {}
                if reference:
                    cache = caches[i]
                    assert isinstance(cache, KVCache)
                    # Transformer.project_to_logits emits one-token decode queries.
                    mask = causal_chunk_mask(
                        torch.empty(1, 2, 3, dtype=x.dtype),
                        torch.empty(cache.seen + 1, 2, 3, dtype=x.dtype),
                    )
                    extra = {"is_causal": False, "attn_mask": mask}
                x, caches[i] = block.forward_cached(x, cache=caches[i], **extra)
            logits = model.project_to_logits(x)
    return tokens


def _attend_to_memory(module: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
    """Attend the first input's queries to the second input as memory."""
    assert isinstance(module, Attention)
    return module(inputs[0], memory=inputs[1])


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
