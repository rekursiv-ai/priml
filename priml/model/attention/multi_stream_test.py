"""Tests for attention module."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from configgle import PartialConfig
from configgle.testing import assert_pprint_golden
from torch import Tensor

import pytest
import torch

from priml.model.attention.kvcache import (
    KVCache,  # used in preallocated cache test
)
from priml.model.attention.multi_stream import MultiStreamAttention
from priml.model.attention.rope import RoPE
from priml.model.norm import RMSNorm
from priml.testing.bfb import assert_bfb_against_golden, bfb_devices
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


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
    streams = config.make()(list(torch.randn(2, 2, 8, 64)))
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
        golden_dir=_TESTDATA,
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
