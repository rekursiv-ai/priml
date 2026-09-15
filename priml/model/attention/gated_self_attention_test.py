"""Tests for native gated full attention with partial rotary embeddings."""

from __future__ import annotations

import pytest
import torch

from priml.model.attention.gated_self_attention import GatedSelfAttention
from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.kvcache import KVCache
from priml.model.attention.rope import RoPE, RoPEMixed


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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
