"""Tests for kvcache module."""

from __future__ import annotations

import pytest
import torch

from priml.model.kvcache import KVCache
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


def test_kv_cache_basic():
    cache = KVCache.alloc(batch=2, heads=4, max_seq=16, channels_head=8)
    assert cache.length == 0
    k = torch.randn(2, 4, 3, 8)
    v = torch.randn(2, 4, 3, 8)
    k_out, _v_out = cache.update(k, v)
    assert k_out.shape == (2, 4, 3, 8)
    assert cache.length == 3


def test_kv_cache_fifo_rolling():
    cache = KVCache.alloc(batch=1, heads=1, max_seq=4, channels_head=2)
    for i in range(4):
        k = torch.full((1, 1, 1, 2), float(i))
        v = torch.full((1, 1, 1, 2), float(i))
        cache.update(k, v)
    assert cache.length == 4
    # One more should FIFO
    k = torch.full((1, 1, 1, 2), 99.0)
    v = torch.full((1, 1, 1, 2), 99.0)
    k_out, _v_out = cache.update(k, v)
    assert cache.length == 4
    assert k_out[0, 0, -1, 0].item() == 99.0
    assert k_out[0, 0, 0, 0].item() == 1.0


def test_kv_cache_freeze():
    cache = KVCache.alloc(batch=1, heads=1, max_seq=8, channels_head=2)
    k = torch.randn(1, 1, 3, 2)
    v = torch.randn(1, 1, 3, 2)
    cache.update(k, v)
    assert cache.length == 3

    frozen = cache.freeze()
    k2 = torch.randn(1, 1, 1, 2)
    v2 = torch.randn(1, 1, 1, 2)
    k_out, _v_out = frozen.update(k2, v2)
    assert frozen.length == 3
    assert k_out.shape == (1, 1, 3, 2)


def test_kv_cache_seen_tracks_absolute_position():
    """``seen`` counts total tokens ever written, even after FIFO eviction.

    Regression for MODEL-007: RoPE offset read ``length`` (capped at
    ``max_seq``), so absolute positions saturated/repeated once the
    cache filled. ``seen`` must keep counting past capacity.
    """
    cache = KVCache.alloc(batch=1, heads=1, max_seq=2, channels_head=2)
    seen: list[int] = []
    for i in range(4):
        seen.append(cache.seen)
        k = torch.full((1, 1, 1, 2), float(i))
        cache.update(k, k)
    assert seen == [0, 1, 2, 3]


def test_kv_cache_freeze_preserves_seen():
    """Freezing a past-capacity cache must keep the monotonic ``seen`` count.

    ``freeze()`` builds a snapshot; the constructor sets ``seen = length`` from
    the (FIFO-capped) length, dropping the true total so RoPE would reuse
    absolute positions after the freeze. The snapshot must carry ``seen``.
    """
    cache = KVCache.alloc(batch=1, heads=1, max_seq=2, channels_head=2)
    for i in range(4):
        cache.update(
            torch.full((1, 1, 1, 2), float(i)), torch.full((1, 1, 1, 2), float(i))
        )
    assert cache.length == 2
    assert cache.seen == 4
    frozen = cache.freeze()
    assert frozen.seen == 4


def test_kv_cache_update_larger_than_capacity_raises():
    """A single update wider than ``max_seq`` is rejected, not silently sliced.

    Regression for MODEL-003: ``length + s - max_seq`` overflowing past
    ``length`` produced a negative ``keep`` and corrupt slices.
    """
    cache = KVCache.alloc(batch=1, heads=1, max_seq=2, channels_head=2)
    k = torch.randn(1, 1, 3, 2)
    with pytest.raises(ValueError, match="exceeds cache capacity"):
        cache.update(k, k)


def test_kv_cache_from_tensors():
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    cache = KVCache(k, v)
    assert cache.length == 8


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
