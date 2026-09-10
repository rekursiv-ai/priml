"""Tests for attention module."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

from configgle import PartialConfig
from configgle.testing import assert_pprint_golden
from torch import Tensor, nn

import pytest
import torch

from priml.model.attention.kvcache import (
    KVCache,  # used in preallocated cache test
)
from priml.model.attention.multi_stream import MultiStreamAttention
from priml.model.attention.rope import RoPE
from priml.model.attention.self_attention import (
    AttentionProjections,
    SelfAttention,
)
from priml.model.norm import RMSNorm
from priml.testing.bfb import (
    assert_bfb_against_golden,
    bfb_devices,
    host_agnostic_numerics,
)
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


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
    """MultiStreamAttention resolves the norm width like SelfAttention does."""
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
        assert isinstance(stream, Tensor)
        assert stream.shape == (2, 8, 64)


def test_multi_stream_norm_out_channels_inferred_from_inner_width():
    """MultiStreamAttention resolves norm_out like SelfAttention does."""
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
    x0 = torch.randn(1, 2, 8)
    x1 = torch.randn(1, 3, 8)
    y0, y1 = m([x0, x1])
    assert isinstance(y0, Tensor)
    assert isinstance(y1, Tensor)
    assert y0.shape == (1, 2, 8)
    assert y1.shape == (1, 3, 8)


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
    assert isinstance(y, Tensor)
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
    assert isinstance(y0, Tensor)
    assert isinstance(y1, Tensor)
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)


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
    assert isinstance(y0, Tensor)
    assert isinstance(y1, Tensor)
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
    assert isinstance(y0, Tensor)
    assert isinstance(y1, Tensor)
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
    assert isinstance(outputs, tuple)
    assert isinstance(caches, list)
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

    outputs, caches = m.forward_cached([torch.randn(1, 3, 8)], cache=[None])

    assert outputs[0].shape == (1, 3, 8)
    assert caches[0].length == 3


def test_multi_stream_no_cache_returns_tuple():
    """Without cache kwarg, returns plain tuple of tensors."""
    m = MultiStreamAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_head=16,
        num_streams=2,
    ).make()
    y0, y1 = m([torch.randn(2, 8, 64), torch.randn(2, 12, 64)])
    assert isinstance(y0, Tensor)
    assert isinstance(y1, Tensor)


def test_multi_stream_causal_requires_single_stream():
    with pytest.raises(ValueError, match="causal=True requires num_streams=1"):
        MultiStreamAttention.Config(
            channels_in=64,
            num_heads=4,
            channels_head=16,
            num_streams=2,
            causal=True,
        ).make()


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
    assert isinstance(y0, Tensor)
    assert isinstance(y1, Tensor)
    assert y0.shape == (2, 8, 64)
    assert y1.shape == (2, 12, 64)


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
    x = torch.randn(1, 4, 16)

    attention((x, x), message=message)

    assert messages == [message, message]


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_multi_stream_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="multi_stream_attention",
        build_module=lambda: (
            MultiStreamAttention.Config(
                channels_in=16,
                num_heads=2,
                channels_head=8,
                num_streams=2,
            )
            .make()
            .to(device)
        ),
        build_input=lambda: [torch.randn(2, 3, 16), torch.randn(2, 4, 16)],
        seed=0,
        run=lambda module, xs: torch.cat(cast(tuple[Tensor, ...], module(xs)), dim=1),
    )


def test_explicit_streams_own_norms_and_native_weights() -> None:
    source = SelfAttention.Config()
    source.channels_in = 8
    source.num_heads = 2
    source.num_heads_kv = 1
    source.channels_head = 4
    source.norm_qk = RMSNorm.Config()
    source.norm_qk.elementwise_affine = True
    source.share_qk_norm = False
    source.norm_out = RMSNorm.Config()
    source.norm_out.elementwise_affine = True
    cfg = MultiStreamAttention.Config()
    cfg.channels_in = 8
    cfg.num_heads = 2
    cfg.num_heads_kv = 1
    cfg.channels_head = 4
    cfg.streams = [source.copy_tree(), source.copy_tree()]
    model = cfg.make()
    native = source.make()
    model.load_stream(0, source=native)
    model.load_stream(1, source=model.streams[0])
    assert model.streams[0].norm_q is not model.streams[1].norm_q
    assert model.streams[0].norm_q is not model.streams[0].norm_k
    x = torch.randn(1, 2, 8)
    other = torch.randn(1, 3, 8)
    masks = [
        torch.cat((torch.zeros(2, 2), torch.full((2, 3), float("-inf"))), -1),
        torch.cat((torch.full((3, 2), float("-inf")), torch.zeros(3, 3)), -1),
    ]
    with host_agnostic_numerics():
        actual = model([x, other], attn_mask=masks)
        assert torch.equal(actual[0], native(x))
        assert torch.equal(actual[1], native(other))
    assert all(
        torch.equal(value, native.state_dict()[key])
        for key, value in model.streams[0].state_dict().items()
    )


def test_per_query_masks_isolate_unequal_streams() -> None:
    cfg = MultiStreamAttention.Config()
    cfg.channels_in = 8
    cfg.num_heads = 2
    model = cfg.make()
    x, other = torch.randn(1, 2, 8), torch.randn(1, 3, 8)
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
    xs = [torch.randn(1, 2, 8), torch.randn(1, 3, 8)]
    actual = model(xs, dropout_p=0.0)
    expected = model.eval()(xs)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


@pytest.mark.parametrize("index", [-1, 1])
def test_attention_loading_rejects_invalid_indices(index: int) -> None:
    cfg = MultiStreamAttention.Config()
    cfg.channels_in = 8
    cfg.num_heads = 2
    cfg.streams = [SelfAttention.Config()]
    source = SelfAttention.Config(channels_in=8, num_heads=2).make()
    with pytest.raises(ValueError, match="index"):
        cfg.make().load_stream(index, source=source)


@pytest.mark.parametrize("explicit", [False, True])
def test_width_mismatch_is_rejected_only_when_building(explicit: bool) -> None:
    cfg = MultiStreamAttention.Config()
    cfg.channels_in = 8
    cfg.channels_out = 12
    cfg.num_heads = 2
    if explicit:
        cfg.streams = [AttentionProjections.Config()]
    finalized = cfg.copy_tree().finalize()
    assert finalized.channels_in == 8
    assert finalized.channels_out == 12
    assert finalized.channels_head == 4
    with pytest.raises(ValueError, match="MultiStreamAttention"):
        cfg.make()
    with pytest.raises(ValueError, match="MultiStreamAttention"):
        MultiStreamAttention(finalized)


@pytest.mark.parametrize("channels", [-1, 7])
def test_joint_attention_rejects_unresolved_geometry(channels: int) -> None:
    cfg = MultiStreamAttention.Config()
    cfg.channels_in = channels
    cfg.num_heads = 2
    with pytest.raises(ValueError, match=r"Need at least two|not divisible"):
        cfg.make()


def test_native_loading_rejects_source_kernel_state_without_partial_copy() -> None:
    source_cfg = SelfAttention.Config()
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
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="state keys"):
        model.load_stream(0, source=source)
    assert all(
        torch.equal(before[name], value) for name, value in model.state_dict().items()
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
