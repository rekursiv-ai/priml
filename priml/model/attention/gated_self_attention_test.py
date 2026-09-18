"""Tests for native gated full attention with partial rotary embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

import pytest
import torch

from priml.cost import cost
from priml.model.attention.gated_self_attention import GatedSelfAttention
from priml.model.attention.kernel import (
    SdpaFused,
    SdpaNaive,
    attention_kernel_cost,
)
from priml.model.attention.kvcache import KVCache
from priml.model.attention.rope import RoPE, RoPEMixed, rotation_cost
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from priml.model.custom_types import AttentionKernel


def test_gated_attention_cache_continuation() -> None:
    config = GatedSelfAttention.Config()
    config.channels_in = 8
    config.num_heads = 2
    config.num_heads_kv = 1
    config.channels_head = 4
    config.rope = RoPE.Config(2)
    config.attn_kernel = SdpaNaive.Config()
    model = config.make().eval()
    x = torch.randn(1, 5, 8)
    with torch.no_grad():
        expected = model(x)
        cache = model.alloc_kv_cache(batch=1, max_seq=5)
        prefix, cache = model.forward_cached(x[:, :3], cache=cache)
        suffix, cache = model.forward_cached(x[:, 3:], cache=cache)
    torch.testing.assert_close(torch.cat([prefix, suffix], dim=1), expected)
    assert cache.seen == 5


def test_gated_attention_parameters_receive_gradients() -> None:
    config = GatedSelfAttention.Config()
    config.channels_in = 8
    config.num_heads = 2
    config.num_heads_kv = 1
    config.channels_head = 4
    model = config.make()
    model(torch.randn(1, 3, 8)).square().sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("dropout", [-0.01, float("nan"), 1.01])
def test_gated_attention_rejects_invalid_dropout(dropout: float) -> None:
    config = GatedSelfAttention.Config()
    config.channels_in = 4
    config.num_heads = config.num_heads_kv = 1
    config.channels_head = 4
    config.dropout = dropout

    with pytest.raises(ValueError, match="dropout"):
        config.make()


def test_gated_attention_reset_resets_injected_rope() -> None:
    config = GatedSelfAttention.Config()
    config.channels_in = 4
    config.num_heads = 2
    config.num_heads_kv = 1
    config.channels_head = 4
    rope = RoPEMixed.Config([2, 2])
    rope.num_heads = 2
    rope.learnable = True
    config.rope = rope
    model = config.make()
    assert isinstance(model.rope, RoPEMixed)
    frequency = next(model.rope.parameters())
    with torch.no_grad():
        frequency.zero_()

    model.reset_parameters()

    assert torch.count_nonzero(frequency)


@pytest.mark.parametrize(
    ("batch", "num_heads", "channels_head"),
    [(2, 1, 4), (1, 2, 4), (1, 1, 5)],
)
def test_gated_attention_rejects_cache_geometry_before_mutation(
    batch: int,
    num_heads: int,
    channels_head: int,
) -> None:
    config = GatedSelfAttention.Config()
    config.channels_in = 8
    config.num_heads = 2
    config.num_heads_kv = 1
    config.channels_head = 4
    model = config.make()
    cache = KVCache.alloc(
        batch=batch,
        num_heads=num_heads,
        max_seq=2,
        channels_head=channels_head,
    )
    original_k, original_v = cache.k.clone(), cache.v.clone()

    with pytest.raises(ValueError, match="cache batch, head, and feature geometry"):
        model.forward_cached(torch.randn(1, 1, 8), cache=cache)

    assert cache.length == 0
    assert cache.seen == 0
    assert torch.equal(cache.k, original_k)
    assert torch.equal(cache.v, original_v)


def test_gated_attention_rejects_rotary_width_larger_than_head() -> None:
    config = GatedSelfAttention.Config()
    config.channels_in = 8
    config.num_heads = config.num_heads_kv = 1
    config.channels_head = 4
    rope = RoPE.Config()
    rope.channels_head = 6
    config.rope = rope
    model = config.make()

    with pytest.raises(ValueError, match="Rotary width"):
        model(torch.randn(1, 1, 8))


@pytest.mark.parametrize(
    "positions",
    [torch.arange(3).reshape(1, 3), torch.arange(6).reshape(1, 3, 2)],
)
def test_gated_attention_rejects_non_text_position_layout(
    positions: torch.Tensor,
) -> None:
    config = GatedSelfAttention.Config()
    config.channels_in = 8
    config.num_heads = config.num_heads_kv = 1
    config.channels_head = 4
    rope = RoPE.Config()
    rope.channels_head = 2
    config.rope = rope
    model = config.make()

    with pytest.raises(ValueError, match="positions must be a text layout"):
        model(torch.randn(1, 3, 8), positions=positions)


def test_gated_attention_accepts_axis_last_text_positions() -> None:
    config = GatedSelfAttention.Config()
    config.channels_in = 8
    config.num_heads = config.num_heads_kv = 1
    config.channels_head = 4
    rope = RoPE.Config()
    rope.channels_head = 2
    config.rope = rope
    model = config.make()
    x = torch.randn(1, 3, 8)

    expected = model(x)
    actual = model(x, positions=torch.arange(3).reshape(1, 3, 1))

    assert torch.equal(actual, expected)


class _SpyKernel(nn.Module):
    """Wrap a kernel and record the kwargs it last received."""

    def __init__(self, inner: AttentionKernel) -> None:
        super().__init__()
        self.inner = inner
        self.received: dict[str, object] = {}

    @override
    def forward(self, q: Tensor, k: Tensor, v: Tensor, **kwargs: object) -> Tensor:
        self.received = kwargs
        return self.inner(q, k, v, **kwargs)


def test_gated_attention_window_restricts_attention() -> None:
    """A caller requesting window=k must get windowed, not full, attention.

    Regression for a bug where the kernel's window argument was silently
    discarded by a hand-rolled full-causal mask (trax Issue#20643).
    """
    config = GatedSelfAttention.Config()
    config.channels_in = 8
    config.num_heads = config.num_heads_kv = 1
    config.channels_head = 4
    model = config.make().eval()
    x = torch.randn(1, 8, 8)

    with torch.no_grad():
        windowed = model(x, window=2)
        full = model(x)

    assert not torch.equal(windowed, full)


@pytest.mark.parametrize("kernel_config", [SdpaNaive.Config(), SdpaFused.Config()])
def test_gated_attention_cached_decode_window_restricts_attention(
    kernel_config: SdpaNaive.Config | SdpaFused.Config,
) -> None:
    """window= must still restrict attention once a cache makes q and k non-square.

    Regression for a third silent-drop of the same bug: the no-explicit-mask
    branch always built a rectangular ``causal_chunk_mask``, bypassing the
    kernel's own window construction.
    """
    config = GatedSelfAttention.Config()
    config.channels_in = 8
    config.num_heads = config.num_heads_kv = 1
    config.channels_head = 4
    config.attn_kernel = kernel_config
    model = config.make().eval()
    x = torch.randn(1, 6, 8)

    with torch.no_grad():
        cache_a = model.alloc_kv_cache(batch=1, max_seq=6)
        cache_b = model.alloc_kv_cache(batch=1, max_seq=6)
        _, cache_a = model.forward_cached(x[:, :4], cache=cache_a)
        _, cache_b = model.forward_cached(x[:, :4], cache=cache_b)
        windowed, _ = model.forward_cached(x[:, 4:], cache=cache_a, window=0)
        full, _ = model.forward_cached(x[:, 4:], cache=cache_b, window=-1)

    assert not torch.equal(windowed, full)


def test_gated_attention_cached_decode_window_zero_pins_to_value_projection() -> None:
    """window=0 at a cached decode step admits only the new token's own key.

    Softmax over one unmasked key is always weight 1, regardless of how many
    tokens precede it in the cache -- the same closed form as the square
    window=0 case, now proven to hold across a cache boundary.
    """
    config = GatedSelfAttention.Config()
    config.channels_in = 8
    config.num_heads = config.num_heads_kv = 1
    config.channels_head = 4
    model = config.make().eval()
    x = torch.randn(1, 6, 8)
    decode = x[:, 4:]

    with torch.no_grad():
        cache = model.alloc_kv_cache(batch=1, max_seq=6)
        _, cache = model.forward_cached(x[:, :4], cache=cache)
        actual, _ = model.forward_cached(decode, cache=cache, window=0)

        shape = (*decode.shape[:-1], -1, model.channels_head)
        gate = (
            model.proj_q(decode)
            .reshape(*decode.shape[:-1], model.num_heads, 2 * model.channels_head)
            .chunk(2, dim=-1)[1]
        )
        v = model.proj_v(decode).reshape(shape)
        expected = model.proj_out((v * gate.sigmoid()).flatten(-2))

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("kernel_config", [SdpaNaive.Config(), SdpaFused.Config()])
@pytest.mark.parametrize(
    ("window", "expected"),
    [
        (0, torch.tensor([[[3.0], [4.0]]])),
        (3, torch.tensor([[[2.0], [2.5]]])),
        (4, torch.tensor([[[2.0], [2.5]]])),
        (5, torch.tensor([[[2.0], [2.5]]])),
        (-1, torch.tensor([[[2.0], [2.5]]])),
    ],
)
def test_gated_attention_cached_chunk_keeps_causality_when_window_is_unmasked(
    kernel_config: SdpaNaive.Config | SdpaFused.Config,
    window: int,
    expected: Tensor,
) -> None:
    """A no-op window must retain causal masking across a cached chunk."""
    config = GatedSelfAttention.Config()
    config.channels_in = 1
    config.num_heads = config.num_heads_kv = config.channels_head = 1
    config.attn_kernel = kernel_config
    model = config.make().eval()
    with torch.no_grad():
        model.proj_q.weight.zero_()
        model.proj_k.weight.zero_()
        model.proj_v.weight.fill_(1)
        model.proj_out.weight.fill_(1)
        cache = model.alloc_kv_cache(batch=1, max_seq=4)
        _, cache = model.forward_cached(torch.tensor([[[2.0], [4.0]]]), cache=cache)
        actual, _ = model.forward_cached(
            torch.tensor([[[6.0], [8.0]]]),
            cache=cache,
            window=window,
        )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("is_causal", [False, True])
def test_gated_attention_consumes_caller_causality_message(is_causal: bool) -> None:
    """Caller causality messages cannot override this causal attention module."""
    config = GatedSelfAttention.Config()
    config.channels_in = 4
    config.num_heads = config.num_heads_kv = 1
    config.channels_head = 4
    model = config.make().eval()
    x = torch.randn(1, 4, 4)

    with torch.no_grad():
        expected = model(x)
        direct = model(x, is_causal=is_causal)
        cache = model.alloc_kv_cache(batch=1, max_seq=4)
        prefix, cache = model.forward_cached(x[:, :2], cache=cache, is_causal=is_causal)
        suffix, _ = model.forward_cached(x[:, 2:], cache=cache, is_causal=is_causal)

    torch.testing.assert_close(direct, expected)
    torch.testing.assert_close(torch.cat((prefix, suffix), dim=1), expected)


def test_gated_attention_window_applies_alongside_an_explicit_mask() -> None:
    """window= must still apply when the caller also supplies an attn_mask.

    Regression for a narrower survival of the same bug: honouring window in
    the no-mask branch left the explicit-mask branch (e.g. Qwen 3.5's
    padding path) still silently discarding it.
    """
    config = GatedSelfAttention.Config()
    config.channels_in = 8
    config.num_heads = config.num_heads_kv = 1
    config.channels_head = 4
    model = config.make().eval()
    x = torch.randn(1, 8, 8)
    attn_mask = torch.zeros(1, 1, 8, 8)

    with torch.no_grad():
        windowed = model(x, window=2, attn_mask=attn_mask)
        full = model(x, attn_mask=attn_mask)

    assert not torch.equal(windowed, full)


def test_gated_attention_window_zero_with_a_no_op_mask_pins_to_value_projection() -> (
    None
):
    """The window+mask combination is correct, not merely present.

    A no-op (all-zero) explicit mask adds nothing, so window=0 combined
    with one must still collapse to the same closed form as window=0
    alone -- an independent check that the combination is additive, not
    that a caller-supplied mask silently wins again.
    """
    config = GatedSelfAttention.Config()
    config.channels_in = 8
    config.num_heads = config.num_heads_kv = 1
    config.channels_head = 4
    model = config.make().eval()
    x = torch.randn(1, 6, 8)
    attn_mask = torch.zeros(1, 1, 6, 6)

    with torch.no_grad():
        actual = model(x, window=0, attn_mask=attn_mask)
        shape = (*x.shape[:-1], -1, model.channels_head)
        gate = (
            model.proj_q(x)
            .reshape(*x.shape[:-1], model.num_heads, 2 * model.channels_head)
            .chunk(2, dim=-1)[1]
        )
        v = model.proj_v(x).reshape(shape)
        expected = model.proj_out((v * gate.sigmoid()).flatten(-2))

    torch.testing.assert_close(actual, expected)


def test_gated_attention_window_zero_pins_to_value_projection() -> None:
    """window=0 admits only each query's own position.

    Softmax over exactly one unmasked key is always weight 1, so the
    attention output collapses to the (gated) value projection -- an
    oracle derived independently of the windowing/masking machinery under
    test.
    """
    config = GatedSelfAttention.Config()
    config.channels_in = 8
    config.num_heads = config.num_heads_kv = 1
    config.channels_head = 4
    model = config.make().eval()
    x = torch.randn(1, 6, 8)

    with torch.no_grad():
        actual = model(x, window=0)
        shape = (*x.shape[:-1], -1, model.channels_head)
        gate = (
            model.proj_q(x)
            .reshape(*x.shape[:-1], model.num_heads, 2 * model.channels_head)
            .chunk(2, dim=-1)[1]
        )
        v = model.proj_v(x).reshape(shape)
        expected = model.proj_out((v * gate.sigmoid()).flatten(-2))

    torch.testing.assert_close(actual, expected)


def test_gated_attention_reaches_causal_fast_path() -> None:
    """The square, unmasked case must dispatch is_causal with no attn_mask.

    A hand-rolled additive mask permanently disables SDPA's causal fast path
    (kernel.py gates it on ``is_causal and attn_mask is None``).
    """
    config = GatedSelfAttention.Config()
    config.channels_in = 8
    config.num_heads = config.num_heads_kv = 1
    config.channels_head = 4
    model = config.make().eval()
    spy = _SpyKernel(model.attn_kernel)
    model.attn_kernel = spy

    with torch.no_grad():
        model(torch.randn(1, 5, 8))

    assert spy.received["attn_mask"] is None
    assert spy.received["is_causal"] is True


def test_gated_attention_cost_is_projections_norms_rotary_kernel_and_gate() -> None:
    """Four projections and the naive kernel are torch's whole matmul count.

    The query projection is twice the query width because it emits the gate
    beside the queries; the gate itself is scalar work, so the elementwise
    silo decomposes into the two norms (each over its own head rows), the
    rotary factors and rotation, the kernel's softmax, and a sigmoid plus a
    product per inner channel.
    """
    config = GatedSelfAttention.Config()
    config.channels_in = 16
    config.num_heads = 2
    config.num_heads_kv = 1
    config.channels_head = 8
    config.rope = RoPE.Config(4)
    model_cost = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(1, 8, 16, requires_grad=True),
        seq_len=8,
        batch_size=1,
        num_tokens=8,
        dtype=None,
    )
    finalized = config.copy_tree().finalize()
    kernel = attention_kernel_cost(seq_len=8, dtype=None, num_heads=2, channels_head=8)
    inner = 2 * 8
    projections = 16 * 2 * inner + 2 * 16 * 8 + inner * 16
    norms = 2 * 8  # norm_q and norm_k each own one head-width scale.
    assert model_cost.params == projections + norms
    assert (
        model_cost["flops", "primal", "matmul"].sum()
        == 2 * projections + kernel["flops", "primal", "matmul"].sum()
    )
    assert model_cost["flops", "adjoint", "matmul"].sum() == (
        4 * projections + kernel["flops", "adjoint", "matmul"].sum()
    )
    assert model_cost.bytes_state == 4 * 2 * 1 * 8
    assert finalized.rope is not None
    scalar = (
        kernel
        + cost(finalized.norm_qk, seq_len=8 * 2, batch_size=1, dtype=None).tile(2)
        + cost(finalized.norm_qk, seq_len=8 * 1, batch_size=1, dtype=None).tile(1)
        + cost(finalized.rope, seq_len=8, batch_size=1, dtype=None)
        + rotation_cost(finalized.rope, rows=8, dtype=None, channels_head=8, heads=3)
    )
    assert model_cost["flops", "primal", "elementwise"].sum() == (
        scalar["flops", "primal", "elementwise"].sum() + 5 * inner
    )
    assert model_cost["flops", "adjoint", "elementwise"].sum() == (
        scalar["flops", "adjoint", "elementwise"].sum() + 6 * inner
    )


def test_gated_attention_cost_hands_dropout_to_the_kernel() -> None:
    """Attention dropout is a mask and a rescale over each head's key row, both ways."""
    config = GatedSelfAttention.Config()
    config.channels_in = 16
    config.num_heads = 2
    config.num_heads_kv = 1
    config.channels_head = 8
    dry = config.copy_tree().finalize().cost(seq_len=32, batch_size=1, dtype=None)
    config.dropout = 0.1
    wet = config.copy_tree().finalize().cost(seq_len=32, batch_size=1, dtype=None)
    assert (
        wet["flops", "elementwise"].sum() - dry["flops", "elementwise"].sum()
        == 2 * 4 * 32
    )
    assert wet["flops", "matmul"].sum() == dry["flops", "matmul"].sum()


def test_gated_attention_traffic_propagates_itemsize() -> None:
    config = GatedSelfAttention.Config()
    config.channels_in = 8
    config.num_heads = 2
    config.num_heads_kv = 1
    config.channels_head = 4
    config.rope = RoPE.Config(2)
    config = config.copy_tree().finalize()
    small = config.cost(seq_len=8, batch_size=1, dtype=torch.bfloat16)
    large = config.cost(seq_len=8, batch_size=1, dtype=None)
    assert (
        large["bytes", torch.float32].sum() == small["bytes", torch.bfloat16].sum() * 2
    )
    assert small.bytes_state == 2 * 2 * 1 * 4


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
