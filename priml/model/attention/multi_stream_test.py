"""Tests for attention module."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Final, cast
from unittest.mock import Mock

from configgle import PartialConfig
from configgle.testing import assert_pprint_golden
from torch import Tensor, nn

import pytest
import torch

from priml.cost import Cost, cost
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
from priml.model.norm import RMSNorm
from priml.testing.bfb import (
    assert_bfb_against_golden,
    bfb_devices,
    host_agnostic_numerics,
)
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


def test_multi_stream_config_pprint() -> None:
    config = MultiStreamAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        num_streams=2,
    )
    assert_pprint_golden(
        test_file=__file__,
        name="multi_stream_attention",
        config=config,
    )


def test_multi_stream_norm_qk_channels_inferred_from_channels_head():
    """MultiStreamAttention resolves the norm width like Attention does."""
    config = MultiStreamAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        norm_qk=RMSNorm.Config(),
    ).finalize()

    assert isinstance(config.norm_qk, RMSNorm.Config)
    assert config.norm_qk.channels_in == 16
    streams = config.make()([torch.randn(2, 8, 64), torch.randn(2, 8, 64)])
    for stream in streams:
        assert stream.shape == (2, 8, 64)


def test_multi_stream_norm_out_channels_inferred_from_inner_width():
    """MultiStreamAttention resolves norm_out like Attention does."""
    config = MultiStreamAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=32,
        norm_out=RMSNorm.Config(),
    ).finalize()

    assert isinstance(config.norm_out, RMSNorm.Config)
    assert config.norm_out.channels_in == 128


def test_multi_stream_2_streams():
    m = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        num_streams=2,
    ).make()
    x0 = torch.randn(4, 2, 8)
    x1 = torch.randn(4, 3, 8)
    y0, y1 = m([x0, x1])
    assert y0.shape == (4, 2, 8)
    assert y1.shape == (4, 3, 8)


def test_multi_stream_causal_single_stream_is_allowed() -> None:
    attention = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        num_streams=1,
        causal=True,
    ).make()
    assert attention.causal


def test_multi_stream_1_stream():
    m = MultiStreamAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        num_streams=1,
    ).make()
    x = torch.randn(2, 16, 64)
    result = m([x])
    assert len(result) == 1
    y = result[0]
    assert y.shape == (2, 16, 64)


def test_multi_stream_gqa():
    m = MultiStreamAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        num_heads_kv=2,
        num_streams=2,
    ).make()
    x0 = torch.randn(2, 8, 64)
    x1 = torch.randn(2, 12, 64)
    y0, y1 = m([x0, x1])
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)


def test_multi_stream_projection_configuration_is_preserved() -> None:
    depth_index = ((1, 2),)
    attention = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        num_heads_kv=1,
        channels_head=4,
        bias=True,
        depth_index=depth_index,
    ).make()

    assert attention.depth_index == depth_index
    assert attention.channels_head == 4
    assert attention.num_heads_kv == 1
    assert attention.kv_groups == 2
    assert isinstance(attention.streams, nn.ModuleList)
    assert attention.proj_qkvs[0].weight.shape == (4, 4, 8)
    assert attention.proj_qkvs[0].bias is not None
    assert attention.proj_qkvs[0].bias.shape == (4, 4)
    assert attention.proj_qkvs[0].depth_index == depth_index
    assert attention.proj_qkvs[0].shard == "colwise"
    assert attention.proj_outs[0].weight.shape == (8, 8)
    assert attention.proj_outs[0].bias is not None
    assert attention.proj_outs[0].depth_index == depth_index
    assert attention.proj_outs[0].shard == "rowwise"


def test_multi_stream_init_weight_reaches_both_projection_groups() -> None:
    def init_ones(weight: Tensor) -> None:
        nn.init.ones_(weight)

    attention = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        init_weight=init_ones,
    ).make()

    assert torch.all(attention.proj_qkvs[0].weight == 1)
    assert torch.all(attention.proj_outs[0].weight == 1)


def test_multi_stream_rejects_unresolved_head_geometry() -> None:
    with pytest.raises(
        ValueError,
        match=r"channels_in=8 not divisible by num_heads=3",
    ):
        MultiStreamAttention.Config(
            channels_in=8,
            num_heads=3,
            channels_head=-1,
        ).make()


def test_multi_stream_uncausal_single_stream_stays_uncausal() -> None:
    causal_flags: list[bool] = []

    def kernel(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        is_causal: bool,
        **kwargs: object,
    ) -> Tensor:
        del k, v, kwargs
        causal_flags.append(is_causal)
        return q

    attention = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        num_streams=1,
        attn_kernel=PartialConfig(kernel),
    ).make()
    attention([torch.randn(2, 3, 8)])

    assert causal_flags == [False]


def test_multi_stream_with_rope():
    rope = RoPE.Config(channels_head=16).make()
    m = MultiStreamAttention.Config(
        channels_in=64,
        num_heads=4,
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
        num_heads=4,
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
        num_heads=4,
        channels_head=16,
        num_streams=2,
        rope=[RoPE.Config(channels_head=16)],
        norm_qk=RMSNorm.Config(),
        norm_out=RMSNorm.Config(),
        share_qk_norm=False,
    ).make()
    m.reset_parameters()


def test_shared_qk_norm_resets_once(monkeypatch: pytest.MonkeyPatch) -> None:
    model = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        norm_qk=RMSNorm.Config(elementwise_affine=True),
    ).make()
    assert model.norm_q is model.norm_k
    assert model.norm_q is not None
    reset = Mock(wraps=model.norm_q.reset_parameters)
    monkeypatch.setattr(model.norm_q, "reset_parameters", reset)

    model.reset_parameters()

    reset.assert_called_once_with()


def test_independent_qk_norms_both_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    model = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        norm_qk=RMSNorm.Config(elementwise_affine=True),
        share_qk_norm=False,
    ).make()
    assert model.norm_q is not None
    assert model.norm_k is not None
    reset_q = Mock(wraps=model.norm_q.reset_parameters)
    reset_k = Mock(wraps=model.norm_k.reset_parameters)
    monkeypatch.setattr(model.norm_q, "reset_parameters", reset_q)
    monkeypatch.setattr(model.norm_k, "reset_parameters", reset_k)

    model.reset_parameters()

    reset_q.assert_called_once_with()
    reset_k.assert_called_once_with()


def test_multi_stream_cache():
    m = MultiStreamAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        num_streams=2,
    ).make()
    caches = [
        KVCache.alloc(batch=2, num_heads=4, max_seq=32, channels_head=16),
        KVCache.alloc(batch=2, num_heads=4, max_seq=32, channels_head=16),
    ]
    x0 = torch.randn(2, 8, 64)
    x1 = torch.randn(2, 12, 64)
    result = m.forward_cached([x0, x1], cache=caches)
    assert len(result) == 2
    outputs, caches = result
    y0, y1 = outputs
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)
    assert caches[0].length == 8
    assert caches[1].length == 12


def test_multi_stream_cache_allocates_for_an_uncached_stream() -> None:
    m = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        num_streams=1,
    ).make()

    outputs, caches = m.forward_cached([torch.randn(4, 3, 8)], cache=[None])

    assert outputs[0].shape == (4, 3, 8)
    assert caches[0].length == 3
    assert caches[0].k.shape == (4, 2, 3, 4)
    assert caches[0].v.shape == (4, 2, 3, 4)


def test_forward_cached_forwards_overrides_and_messages() -> None:
    received: list[tuple[float, bool, Tensor | None, object]] = []

    def kernel(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        dropout_p: float,
        is_causal: bool,
        attn_mask: Tensor | None,
        message: object,
    ) -> Tensor:
        del k, v
        received.append((dropout_p, is_causal, attn_mask, message))
        return q

    model = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        rope=[RoPE.Config(4), RoPE.Config(4)],
        attn_kernel=PartialConfig(kernel),
    ).make()
    xs = [torch.randn(2, 4, 8), torch.randn(2, 3, 8)]
    # RoPE.rotate requires a singleton head axis for broadcast factors.
    cos_sin = [
        (torch.ones(2, 1, 2), torch.zeros(2, 1, 2)),
        (torch.ones(3, 1, 2), torch.zeros(3, 1, 2)),
    ]
    masks = [torch.ones(2, 5, dtype=torch.bool), None]
    message = object()

    outputs, caches = model.forward_cached(
        xs,
        cache=[None, None],
        positions=[torch.arange(4), torch.arange(3)],
        cos_sin=cos_sin,
        dropout_p=0.25,
        is_causal=True,
        attn_mask=masks,
        message=message,
    )

    assert [output.shape for output in outputs] == [(2, 4, 8), (2, 3, 8)]
    assert [cache.length for cache in caches] == [4, 3]
    assert [(rate, causal, msg) for rate, causal, _, msg in received] == [
        (0.25, True, message),
        (0.25, True, message),
    ]
    assert received[0][2] is masks[0]
    assert received[1][2] is None
    expected = model(
        xs,
        cos_sin=cos_sin,
        dropout_p=0.25,
        is_causal=True,
        attn_mask=masks,
        message=message,
    )
    assert all(
        torch.equal(actual, target)
        for actual, target in zip(outputs, expected, strict=True)
    )

    positions = [torch.tensor([2, 4]), torch.tensor([1, 5, 8])]
    cached_positions, _ = model.forward_cached(
        xs,
        cache=[None, None],
        positions=positions,
        message=message,
    )
    expected_positions = model(xs, positions=positions, message=message)
    assert all(
        torch.equal(actual, target)
        for actual, target in zip(cached_positions, expected_positions, strict=True)
    )


def test_multi_stream_no_cache_returns_tuple():
    """Without cache kwarg, returns plain tuple of tensors."""
    m = MultiStreamAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        num_streams=2,
    ).make()
    y0, y1 = m([torch.randn(2, 8, 64), torch.randn(2, 12, 64)])
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)


def test_multi_stream_preserves_leading_dimensions() -> None:
    def identity_kernel(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        **kwargs: object,
    ) -> Tensor:
        del k, v, kwargs
        return q

    attention = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        num_streams=2,
        attn_kernel=PartialConfig(identity_kernel),
    ).make()
    with torch.no_grad():
        for projection in attention.proj_outs:
            projection.weight.copy_(torch.eye(8))
    xs = [torch.randn(2, 3, 5, 8), torch.randn(2, 3, 5, 8)]

    outputs = attention(xs)

    for output, x, projection in zip(
        outputs,
        xs,
        attention.proj_qkvs,
        strict=True,
    ):
        q = projection(x).split([2, 2, 2], dim=-2)[0]
        torch.testing.assert_close(output, q.flatten(-2))


def test_multi_stream_causal_requires_single_stream():
    with pytest.raises(
        ValueError,
        match=r"^causal=True requires num_streams=1\.$",
    ):
        MultiStreamAttention.Config(
            channels_in=64,
            num_heads=4,
            channels_head=16,
            num_streams=2,
            causal=True,
        ).make()


def test_explicit_causal_streams_require_single_stream():
    one_stream = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        streams=[AttentionProjections.Config(causal=True)],
    )
    one_stream.make()
    config = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        streams=[
            AttentionProjections.Config(causal=True),
            AttentionProjections.Config(),
        ],
    )
    with pytest.raises(
        ValueError,
        match=r"^Causal streams require a single stream; use per-stream masks\.$",
    ) as exc_info:
        config.make()
    assert str(exc_info.value) == (
        "Causal streams require a single stream; use per-stream masks."
    )


def test_multi_stream_kv_heads_validation():
    with pytest.raises(ValueError, match="must be divisible"):
        MultiStreamAttention.Config(
            num_heads=5,
            channels_head=12,
            num_heads_kv=3,
            num_streams=2,
        ).make()


def test_multi_stream_internal_rope():
    """Internal RoPE via config (not external cos_sin)."""
    m = MultiStreamAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        num_streams=2,
        rope=[RoPE.Config(channels_head=16), None],
    ).make()
    x0 = torch.randn(2, 8, 64)
    x1 = torch.randn(2, 12, 64)
    y0, y1 = m([x0, x1])
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)


def test_internal_rope_positions_use_input_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def identity_kernel(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        **kwargs: object,
    ) -> Tensor:
        del k, v, kwargs
        return q

    attention = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        num_streams=2,
        rope=[RoPE.Config(4), RoPE.Config(4)],
        attn_kernel=PartialConfig(identity_kernel),
    ).make()
    original_arange = torch.arange
    devices: list[torch.device | None] = []

    def record_arange(
        start: int,
        end: int,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        devices.append(device)
        return original_arange(start, end, device=device)

    monkeypatch.setattr(torch, "arange", record_arange)
    xs = [torch.randn(2, 3, 8), torch.randn(2, 4, 8)]

    attention(xs)

    assert devices == [x.device for x in xs]


def test_cached_internal_rope_uses_each_cache_offset() -> None:
    def identity_kernel(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        **kwargs: object,
    ) -> Tensor:
        del k, v, kwargs
        return q

    attention = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        num_streams=2,
        rope=[RoPE.Config(4), RoPE.Config(4)],
        attn_kernel=PartialConfig(identity_kernel),
    ).make()
    initial = [torch.randn(2, 4, 8), torch.randn(2, 3, 8)]
    caches = [
        KVCache.alloc(batch=2, num_heads=2, max_seq=12, channels_head=4),
        KVCache.alloc(batch=2, num_heads=2, max_seq=12, channels_head=4),
    ]
    _, caches = attention.forward_cached(initial, cache=caches)
    xs = [torch.randn(2, 4, 8), torch.randn(2, 3, 8)]
    positions = [torch.arange(4, 8), torch.arange(3, 6)]

    cached, updated = attention.forward_cached(xs, cache=caches)
    cos_sin = [
        attention.ropes[str(index)](position)
        for index, position in enumerate(positions)
    ]
    positioned = attention(xs, positions=positions)
    externally_rotated = attention(xs, cos_sin=cos_sin)

    assert [cache.length for cache in updated] == [8, 6]
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(cached, externally_rotated, strict=True)
    )
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(positioned, externally_rotated, strict=True)
    )


def test_multi_stream_internal_rope_uses_sequence_axis_with_leading_dims() -> None:
    def identity_kernel(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        **kwargs: object,
    ) -> Tensor:
        del k, v, kwargs
        return q

    attention = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        num_streams=2,
        rope=[RoPE.Config(4), RoPE.Config(4)],
        attn_kernel=PartialConfig(identity_kernel),
    ).make()
    with torch.no_grad():
        for projection in attention.proj_outs:
            projection.weight.copy_(torch.eye(8))
    xs = [torch.randn(2, 3, 5, 8), torch.randn(2, 3, 6, 8)]

    internally_rotated = attention(xs)
    cos_sin = [
        attention.ropes[str(index)](torch.arange(x.shape[-2]))
        for index, x in enumerate(xs)
    ]
    externally_rotated = attention(xs, cos_sin=cos_sin)

    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(internally_rotated, externally_rotated, strict=True)
    )


def _recording_multi_stream_attention(
    *,
    num_heads: int = 2,
    num_heads_kv: int = 2,
    dropout: float = 0.0,
    causal: bool = False,
) -> tuple[MultiStreamAttention, list[dict[str, object]]]:
    calls: list[dict[str, object]] = []

    def kernel(q: Tensor, k: Tensor, v: Tensor, **kwargs: object) -> Tensor:
        calls.append({"q": q, "k": k, "v": v, **kwargs})
        return q

    config = MultiStreamAttention.Config(
        channels_in=12,
        num_heads=num_heads,
        channels_head=3,
        num_heads_kv=num_heads_kv,
        num_streams=2,
        dropout=dropout,
        causal=causal,
        attn_kernel=PartialConfig(kernel),
    )
    return config.make(), calls


def test_joint_kv_geometry_and_output_width() -> None:
    attention, calls = _recording_multi_stream_attention(
        num_heads=4,
        num_heads_kv=2,
    )
    xs = [torch.randn(2, 3, 12), torch.randn(2, 5, 12)]

    outputs = attention(xs)

    assert [output.shape for output in outputs] == [(2, 3, 12), (2, 5, 12)]
    assert len(calls) == 2
    for call, query_length in zip(calls, (3, 5), strict=True):
        query, key, value = call["q"], call["k"], call["v"]
        assert isinstance(query, Tensor)
        assert isinstance(key, Tensor)
        assert isinstance(value, Tensor)
        assert query.shape == (2, query_length, 4, 3)
        assert key.shape == (2, 8, 4, 3)
        assert value.shape == (2, 8, 4, 3)


def test_training_dropout_eval_and_explicit_override() -> None:
    attention, calls = _recording_multi_stream_attention(dropout=0.25)
    xs = [torch.randn(2, 3, 12), torch.randn(2, 5, 12)]

    attention.train()
    attention(xs)
    attention.eval()
    attention(xs)
    attention.train()
    attention(xs, dropout_p=0.5)

    assert [call["dropout_p"] for call in calls] == [0.25, 0.25, 0.0, 0.0, 0.5, 0.5]


def test_explicit_per_query_masks_and_causal_override_reach_kernel() -> None:
    attention, calls = _recording_multi_stream_attention()
    xs = [torch.randn(2, 3, 12), torch.randn(2, 5, 12)]
    masks = [torch.ones(3, 8, dtype=torch.bool), None]

    attention(xs, attn_mask=masks, is_causal=True)

    assert calls[0]["attn_mask"] is masks[0]
    assert calls[0]["is_causal"] is True
    assert calls[1]["attn_mask"] is None
    assert calls[1]["is_causal"] is True


def test_single_stream_causal_mask_is_constructed_for_missing_mask() -> None:
    calls: list[dict[str, object]] = []

    def kernel(q: Tensor, k: Tensor, v: Tensor, **kwargs: object) -> Tensor:
        del k, v
        calls.append(kwargs)
        return q

    attention = MultiStreamAttention.Config(
        channels_in=12,
        num_heads=2,
        channels_head=3,
        num_streams=1,
        causal=True,
        attn_kernel=PartialConfig(kernel),
    ).make()

    attention([torch.randn(2, 3, 12)])

    assert calls[0]["is_causal"] is True
    assert calls[0]["attn_mask"] is None


def test_explicit_query_and_kv_dimensions_remain_stream_specific() -> None:
    attention, calls = _recording_multi_stream_attention()
    xs = [torch.randn(2, 3, 12), torch.randn(2, 5, 12)]

    attention(xs)

    query_lengths: list[int] = []
    key_lengths: list[int] = []
    for call in calls:
        query, key = call["q"], call["k"]
        assert isinstance(query, Tensor)
        assert isinstance(key, Tensor)
        query_lengths.append(query.shape[-3])
        key_lengths.append(key.shape[-3])
    assert query_lengths == [3, 5]
    assert key_lengths == [8, 8]


def test_multi_stream_attention_concatenates_context_across_leading_dims() -> None:
    def context_kernel(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        **kwargs: object,
    ) -> Tensor:
        del kwargs
        return q + k.mean(dim=-3, keepdim=True) + v.mean(dim=-3, keepdim=True)

    attention = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        num_streams=2,
        attn_kernel=PartialConfig(context_kernel),
    ).make()
    with torch.no_grad():
        for projection in attention.proj_outs:
            projection.weight.copy_(torch.eye(8))
    xs = [torch.randn(2, 3, 5, 8), torch.randn(2, 3, 6, 8)]
    changed = [xs[0], xs[1] + 1]

    outputs = attention(xs)
    changed_outputs = attention(changed)

    assert [output.shape for output in outputs] == [(2, 3, 5, 8), (2, 3, 6, 8)]
    assert not torch.equal(outputs[0], changed_outputs[0])


def test_multistream_attention_forwards_the_open_message_bus() -> None:
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

    attention = MultiStreamAttention.Config(
        channels_in=16,
        num_streams=2,
        num_heads=2,
        channels_head=8,
        attn_kernel=PartialConfig(kernel),
    ).make()
    message = object()
    x = torch.randn(4, 3, 16)

    attention((x, x), message=message)

    assert messages == [message, message]


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_multi_stream_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="multi_stream_attention",
        build_module=lambda: (
            MultiStreamAttention.Config(
                channels_in=4,
                num_heads=2,
                channels_head=2,
                num_streams=2,
            )
            .make()
            .to(device)
        ),
        build_input=lambda: [torch.randn(2, 3, 4), torch.randn(2, 3, 4)],
        seed=0,
        run=lambda module, xs: torch.cat(cast(tuple[Tensor, ...], module(xs)), dim=1),
    )


@pytest.mark.parametrize("explicit_streams", [False, True])
def test_tensor_parallel_fused_attention_rejects_dtensor_weights(
    monkeypatch: pytest.MonkeyPatch,
    explicit_streams: bool,
) -> None:
    config = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        num_streams=2,
        attn_kernel=SdpaFused.Config(),
    )
    if explicit_streams:
        config.streams = [AttentionProjections.Config(), AttentionProjections.Config()]
    attention = config.make()
    monkeypatch.setattr(
        "priml.model.attention.multi_stream.DTensor",
        nn.Parameter,
    )

    with pytest.raises(RuntimeError) as exc_info:
        attention.assert_tensor_parallel_compatible()
    assert str(exc_info.value) == (
        "Tensor parallelism requires a DTensor-compatible attention "
        "kernel; set attn_kernel=SdpaNaive (the fused flash kernel has "
        "no DTensor sharding strategy)."
    )


def test_tensor_parallel_naive_attention_accepts_dtensor_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        num_streams=2,
        attn_kernel=SdpaNaive.Config(),
    ).make()
    monkeypatch.setattr(
        "priml.model.attention.multi_stream.DTensor",
        nn.Parameter,
    )

    attention.assert_tensor_parallel_compatible()


def test_load_stream_rejects_native_attention_with_exact_message() -> None:
    model = MultiStreamAttention.Config(
        channels_in=4,
        num_heads=2,
        channels_head=2,
    ).make()
    source = AttentionProjections.Config(
        channels_in=4,
        num_heads=2,
        channels_head=2,
    ).make()

    with pytest.raises(
        ValueError,
        match=r"^Native loading requires explicit streams\.$",
    ) as exc_info:
        model.load_stream(0, source=source)

    assert str(exc_info.value) == "Native loading requires explicit streams."


@pytest.mark.parametrize("argument", ["attn_mask", "positions", "cos_sin", "cache"])
def test_forward_reports_argument_name_for_wrong_stream_count(argument: str) -> None:
    attention = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        num_streams=2,
    ).make()
    xs = [torch.randn(2, 3, 8), torch.randn(2, 3, 8)]
    message = f"{argument} must contain one entry per stream (2)."
    if argument == "attn_mask":
        invoke = partial(attention, xs, attn_mask=[None])
    elif argument == "positions":
        invoke = partial(attention, xs, positions=[None])
    elif argument == "cos_sin":
        invoke = partial(attention, xs, cos_sin=[None])
    else:
        invoke = partial(attention.forward_cached, xs, cache=[None])

    with pytest.raises(
        ValueError,
        match=rf"^{argument} must contain one entry per stream \(2\)\.$",
    ) as exc_info:
        invoke()

    assert str(exc_info.value) == message


def test_forward_cached_rejects_missing_updated_cache_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MultiStreamAttention.Config(
        channels_in=4,
        num_heads=2,
        channels_head=2,
    ).make()

    def missing_cache_list(
        *args: object,
        **kwargs: object,
    ) -> tuple[tuple[Tensor, ...], None]:
        del args, kwargs
        return ((torch.zeros(2, 3, 4),), None)

    monkeypatch.setattr(model, "_forward", missing_cache_list)
    inputs = [torch.zeros(2, 3, 4)]

    with pytest.raises(
        ValueError,
        match=r"^Expected updated is not None\.$",
    ) as exc_info:
        model.forward_cached(inputs, cache=[None])

    assert str(exc_info.value) == "Expected updated is not None."


def test_explicit_stream_uses_its_causal_and_dropout_settings() -> None:
    calls: list[dict[str, object]] = []

    def kernel(q: Tensor, k: Tensor, v: Tensor, **kwargs: object) -> Tensor:
        del k, v
        calls.append(kwargs)
        return q

    stream = AttentionProjections.Config(causal=True, dropout=0.25)
    attention = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        streams=[stream],
        attn_kernel=PartialConfig(kernel),
    ).make()
    xs = [torch.randn(2, 3, 8)]

    attention.train()
    attention(xs)
    attention.eval()
    attention(xs)

    assert [call["is_causal"] for call in calls] == [True, True]
    assert [call["dropout_p"] for call in calls] == [0.25, 0.0]


def test_cached_causal_chunk_builds_mask_for_prefix() -> None:
    calls: list[dict[str, object]] = []

    def kernel(q: Tensor, k: Tensor, v: Tensor, **kwargs: object) -> Tensor:
        del k, v
        calls.append(kwargs)
        return q

    attention = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        num_streams=1,
        causal=True,
        attn_kernel=PartialConfig(kernel),
    ).make()
    cache = KVCache.alloc(batch=2, num_heads=2, max_seq=8, channels_head=4)
    attention.forward_cached([torch.randn(2, 3, 8)], cache=[cache])

    attention.forward_cached([torch.randn(2, 3, 8)], cache=[cache])

    mask = calls[-1]["attn_mask"]
    assert isinstance(mask, Tensor)
    assert mask.shape == (3, 6)
    assert torch.equal(
        mask,
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf")],
                [0.0, 0.0, 0.0, 0.0, 0.0, float("-inf")],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ),
    )


def test_explicit_streams_own_norms_and_native_weights() -> None:
    source = Attention.Config()
    source.channels_in = 8
    source.num_heads = 2
    source.num_heads_kv = 1
    source.channels_head = 4
    source.norm_qk = RMSNorm.Config()
    source.norm_qk.elementwise_affine = True
    source.share_qk_norm = False
    source.split_qkv_projection = True
    source.rope = RoPE.Config(4)
    source.norm_out = RMSNorm.Config()
    source.norm_out.elementwise_affine = True
    cfg = MultiStreamAttention.Config()
    cfg.channels_in = 8
    cfg.num_heads = 2
    cfg.num_heads_kv = 1
    cfg.channels_head = 4
    cfg.streams = [source.copy_tree(), source.copy_tree()]
    model = cfg.make()
    assert model.norm_q is model.norm_k is model.norm_out is None
    native = source.make()
    model.load_stream(0, source=native)
    model.load_stream(1, source=model.streams[0])
    assert model.streams[0].norm_q is not model.streams[1].norm_q
    assert model.streams[0].norm_q is not model.streams[0].norm_k
    x = torch.randn(4, 2, 8)
    other = torch.randn(4, 3, 8)
    # MultiStreamAttention.forward uses each square block for one stream.
    masks = [
        # MultiStreamAttention.forward uses a square self-stream mask block.
        torch.cat((torch.zeros(2, 2), torch.full((2, 3), float("-inf"))), -1),
        # MultiStreamAttention.forward uses a square self-stream mask block.
        torch.cat((torch.full((3, 2), float("-inf")), torch.zeros(3, 3)), -1),
    ]
    with host_agnostic_numerics():
        actual = model([x, other], attn_mask=masks)
        assert torch.equal(actual[0], native(x))
        assert torch.equal(actual[1], native(other))
    native_state = native.state_dict()
    stream_state = model.streams[0].state_dict()
    assert all(
        torch.equal(value, native_state[key]) for key, value in stream_state.items()
    )


def test_per_query_masks_isolate_unequal_streams() -> None:
    cfg = MultiStreamAttention.Config()
    cfg.channels_in = 8
    cfg.num_heads = 2
    model = cfg.make()
    x, other = torch.randn(4, 2, 8), torch.randn(4, 3, 8)
    # MultiStreamAttention.forward uses a square self-stream mask block.
    masks = [
        torch.cat((torch.zeros(2, 2), torch.full((2, 3), float("-inf"))), -1),
        None,
    ]
    first = model([x, other], attn_mask=masks)
    changed = model([x, other + 10], attn_mask=masks)
    assert torch.equal(first[0], changed[0])
    assert not torch.equal(first[1], changed[1])
    with pytest.raises(ValueError, match="attn_mask"):
        model([x, other], attn_mask=[masks[0]])
    with pytest.raises(ValueError, match="streams"):
        model([x])


def test_dropout_override_applies_in_training() -> None:
    cfg = MultiStreamAttention.Config()
    cfg.channels_in = 8
    cfg.num_heads = 2
    cfg.dropout = 0.5
    model = cfg.make().train()
    xs = [torch.randn(4, 2, 8), torch.randn(4, 3, 8)]
    actual = model(xs, dropout_p=0.0)
    expected = model.eval()(xs)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


@pytest.mark.parametrize("index", [-1, 1])
def test_attention_loading_rejects_invalid_indices(index: int) -> None:
    cfg = MultiStreamAttention.Config()
    cfg.channels_in = 8
    cfg.num_heads = 2
    cfg.streams = [Attention.Config()]
    source = Attention.Config(channels_in=8, num_heads=2).make()
    with pytest.raises(
        ValueError,
        match=rf"^Invalid stream index {index}\.$",
    ) as exc_info:
        cfg.make().load_stream(index, source=source)
    assert str(exc_info.value) == f"Invalid stream index {index}."


@pytest.mark.parametrize("explicit", [False, True])
def test_width_mismatch_is_rejected_at_make(explicit: bool) -> None:
    cfg = MultiStreamAttention.Config()
    cfg.channels_in = 8
    cfg.channels_out = 12
    cfg.num_heads = 2
    if explicit:
        cfg.streams = [AttentionProjections.Config()]
    with pytest.raises(ValueError, match="channels_in=8 must equal channels_out=12"):
        cfg.make()


@pytest.mark.parametrize("channels", [-1, 7])
def test_joint_attention_rejects_unresolved_geometry(channels: int) -> None:
    cfg = MultiStreamAttention.Config()
    cfg.channels_in = channels
    cfg.num_heads = 2
    with pytest.raises(ValueError, match=r"Need at least two|not divisible"):
        cfg.make()


def test_native_loading_rejects_source_kernel_state_without_partial_copy() -> None:
    source_cfg = Attention.Config()
    source_cfg.channels_in = 8
    source_cfg.num_heads = 2
    source = source_cfg.make()
    assert isinstance(source.attn_kernel, nn.Module)
    source.attn_kernel.register_buffer("checkpoint_state", torch.ones(1))
    cfg = MultiStreamAttention.Config()
    cfg.channels_in = 8
    cfg.num_heads = 2
    cfg.streams = [AttentionProjections.Config().update(source_cfg, skip_missing=True)]
    model = cfg.make()
    state = model.state_dict()
    before = {name: value.clone() for name, value in state.items()}
    with pytest.raises(
        ValueError,
        match=r"^Native stream state keys do not match the configured destination\.$",
    ) as exc_info:
        model.load_stream(0, source=source)
    assert str(exc_info.value) == (
        "Native stream state keys do not match the configured destination."
    )
    assert all(torch.equal(before[name], value) for name, value in state.items())


def test_native_loading_rejects_shape_mismatch_before_copy() -> None:
    stream_config = AttentionProjections.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
    )
    config = MultiStreamAttention.Config(
        channels_in=8,
        num_heads=2,
        channels_head=4,
        streams=[stream_config],
    )
    model = config.make()
    source = stream_config.make()
    model.streams[0].register_buffer("checkpoint_state", torch.ones(2))
    source.register_buffer("checkpoint_state", torch.ones(3))
    before = model.streams[0].get_buffer("checkpoint_state").clone()

    with pytest.raises(
        ValueError,
        match=r"^Native stream state shape mismatch for checkpoint_state\.$",
    ) as exc_info:
        model.load_stream(0, source=source)

    assert str(exc_info.value) == (
        "Native stream state shape mismatch for checkpoint_state."
    )
    assert torch.equal(model.streams[0].get_buffer("checkpoint_state"), before)


def test_multi_stream_cost_sums_projections_and_scores_per_stream() -> None:
    """Every stream's projections, and each stream's query row over the joint keys."""
    config = MultiStreamAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        num_streams=2,
        rope=[RoPE.Config(channels_head=8), None],
    )
    finalized = config.copy_tree().finalize()
    model_cost = finalized.cost(seq_len=8, batch_size=1, dtype=None)
    # One query row against both streams' keys.
    kernel = attention_kernel_cost(
        seq_len=16,
        rows=8,
        dtype=None,
        num_heads=2,
        channels_head=8,
    )
    stream = (2 + 2 + 2) * 16 * 8 + 16 * 16
    stream_primal = 2 * 8 * stream
    stream_adjoint = 4 * 8 * stream
    assert kernel["flops", "primal", "matmul"].sum() == 4 * 2 * 8 * 16 * 8
    assert model_cost.params == 2 * stream
    assert model_cost.params == sum(p.numel() for p in config.make().parameters())
    assert model_cost["flops", "primal", "matmul"].sum() == 2 * (
        stream_primal + kernel["flops", "primal", "matmul"].sum()
    )
    assert model_cost["flops", "adjoint", "matmul"].sum() == 2 * (
        stream_adjoint + kernel["flops", "adjoint", "matmul"].sum()
    )
    # A position holds one token per stream, each caching its own K and V.
    assert model_cost.bytes_state == 4 * 2 * 2 * 2 * 8


def test_multi_stream_cost_matches_torch_per_stream_token() -> None:
    """Every stream token pays its projections once and attends over joint keys.

    Two streams of four tokens each; the naive kernel makes the attention
    countable. This test caught the kernel being costed once per position
    rather than once per stream token (measured 15,360 to a claimed 13,824).
    """
    config = MultiStreamAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        num_streams=2,
        attn_kernel=SdpaNaive.Config(),
    )

    # ``cost`` costs one position -- every stream's token -- so the token
    # count is one stream's length.
    def run(module: nn.Module, xs: tuple[Tensor, ...]) -> Tensor:
        assert isinstance(module, MultiStreamAttention)
        return torch.stack(module(list(xs))).sum()

    assert_cost_matches_torch(
        config,
        build_input=lambda: tuple(
            torch.randn(2, 4, 16, requires_grad=True) for _ in range(2)
        ),
        seq_len=4,
        batch_size=2,
        dtype=None,
        run=run,
    )


def test_multi_stream_cost_counts_shared_norms_once() -> None:
    config = MultiStreamAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        num_streams=2,
        norm_qk=RMSNorm.Config(elementwise_affine=True),
        norm_out=RMSNorm.Config(elementwise_affine=True),
        share_qk_norm=False,
    )
    cost = config.copy_tree().finalize().cost(seq_len=8, batch_size=1, dtype=None)
    assert cost.params == 2 * ((2 + 2 + 2) * 16 * 8 + 16 * 16) + 2 * 8 + 16
    assert cost.params == sum(p.numel() for p in config.make().parameters())


def test_multi_stream_cost_prices_explicit_streams_by_their_own_config() -> None:
    stream = AttentionProjections.Config(
        norm_qk=RMSNorm.Config(elementwise_affine=True),
        share_qk_norm=False,
        bias=True,
    )
    config = MultiStreamAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_head=8,
        streams=[stream.copy_tree(), stream.copy_tree()],
    )
    finalized = config.copy_tree().finalize()
    model_cost = finalized.cost(seq_len=8, batch_size=1, dtype=None)
    owned = sum(
        (cost(s, seq_len=8, batch_size=1, dtype=None) for s in finalized.streams),
        Cost(),
    )
    kernel = attention_kernel_cost(
        seq_len=16,
        rows=8,
        dtype=None,
        num_heads=2,
        channels_head=8,
    )
    assert model_cost.params == owned.params
    assert model_cost.params == sum(p.numel() for p in config.make().parameters())
    assert model_cost["flops", "primal", "matmul"].sum() == (
        owned["flops", "primal", "matmul"].sum()
        + 2 * kernel["flops", "primal", "matmul"].sum()
    )


def test_multi_stream_traffic_propagates_itemsize_to_rotary_and_projections() -> None:
    config = MultiStreamAttention.Config()
    config.channels_in = 8
    config.num_heads = 2
    config.rope = [RoPE.Config(4)]
    config = config.copy_tree().finalize()
    small = config.cost(seq_len=4, batch_size=1, dtype=torch.bfloat16)
    large = config.cost(seq_len=4, batch_size=1, dtype=None)
    assert (
        large["bytes", torch.float32].sum() == small["bytes", torch.bfloat16].sum() * 2
    )
    assert small.bytes_state == 2 * 2 * 2 * 8


def test_multistream_independent_qk_norm_scales_are_each_read_once() -> None:
    config = MultiStreamAttention.Config()
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
