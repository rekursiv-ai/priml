"""Tests for kvcache module."""

from __future__ import annotations

from pathlib import Path
from typing import Final, override
from unittest.mock import Mock

from torch import Tensor, nn

import pytest
import torch

from priml.model.attention.kvcache import KVCache
from priml.testing.bfb import assert_bfb_against_golden, bfb_devices
from priml.testing.golden import assert_text_golden


_CWD: Final = Path(__file__).resolve().parent


class _Cache(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    @override
    def forward(self, x: Tensor) -> Tensor:
        return _cache_contract(x)


def _cache_contract(x: Tensor) -> Tensor:
    cache = KVCache.alloc(
        batch=x.shape[0],
        num_heads=x.shape[1],
        max_seq=3,
        channels_head=x.shape[-1],
    )
    _ = cache.update(x[..., :2, :], -x[..., :2, :])
    k, v = cache.update(x[..., 2:, :], -x[..., 2:, :])
    frozen = cache.freeze()
    frozen_k, frozen_v = frozen.update(x[..., :1, :], x[..., :1, :])
    metadata = x.new_tensor([cache.length, cache.seen, frozen.length, frozen.seen])
    return torch.cat(
        [k.flatten(), v.flatten(), frozen_k.flatten(), frozen_v.flatten(), metadata],
    )


def test_kv_cache_allocates_requested_shapes_device_and_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zeros = Mock(wraps=torch.zeros)
    monkeypatch.setattr(torch, "zeros", zeros)
    cache = KVCache.alloc(
        batch=(2, 3),
        num_heads=4,
        max_seq=5,
        channels_head=6,
        channels_v_head=7,
        device="cpu",
        dtype=torch.float64,
    )

    assert cache.k.shape == (2, 3, 4, 5, 6)
    assert cache.v.shape == (2, 3, 4, 5, 7)
    assert cache.k.device.type == cache.v.device.type == "cpu"
    assert cache.k.dtype == cache.v.dtype == torch.float64
    assert [call.args for call in zeros.call_args_list] == [
        ((2, 3, 4, 5, 6),),
        ((2, 3, 4, 5, 7),),
    ]
    assert [call.kwargs for call in zeros.call_args_list] == [
        {"device": "cpu", "dtype": torch.float64},
        {"device": "cpu", "dtype": torch.float64},
    ]
    assert torch.count_nonzero(cache.k) == 0
    assert torch.count_nonzero(cache.v) == 0
    assert cache.length == cache.seen == 0


def test_kv_cache_fifo_overflow_by_one_preserves_exact_suffix() -> None:
    cache = KVCache.alloc(batch=2, num_heads=3, max_seq=5, channels_head=4)
    keys, values = cache.update(
        # KVCache inputs broadcast across batch, heads, and channels.
        torch.arange(10, 13, dtype=torch.float32)
        .reshape(1, 1, 3, 1)
        .expand(2, 3, 3, 4),
        -torch.arange(10, 13, dtype=torch.float32)
        .reshape(1, 1, 3, 1)
        .expand(2, 3, 3, 4),
    )
    assert cache.length == 3
    assert cache.seen == 3
    assert keys.shape == values.shape == (2, 3, 3, 4)
    torch.testing.assert_close(keys[0, 0, :, 0], torch.tensor([10.0, 11.0, 12.0]))
    torch.testing.assert_close(values[0, 0, :, 0], torch.tensor([-10.0, -11.0, -12.0]))
    # KVCache inputs broadcast across batch, heads, and channels.
    keys, values = cache.update(
        torch.arange(13, 16, dtype=torch.float32)
        .reshape(1, 1, 3, 1)
        .expand(2, 3, 3, 4),
        -torch.arange(13, 16, dtype=torch.float32)
        .reshape(1, 1, 3, 1)
        .expand(2, 3, 3, 4),
    )

    assert cache.length == 5
    assert cache.seen == 6
    assert keys.shape == values.shape == (2, 3, 5, 4)
    torch.testing.assert_close(
        keys[0, 0, :, 0],
        torch.tensor([11.0, 12.0, 13.0, 14.0, 15.0]),
    )
    torch.testing.assert_close(
        values[0, 0, :, 0],
        torch.tensor([-11.0, -12.0, -13.0, -14.0, -15.0]),
    )


def test_kv_cache_exact_fill_does_not_clone_retained_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = KVCache.alloc(batch=2, num_heads=3, max_seq=5, channels_head=4)
    # KVCache's contract uses equal head and sequence widths here.
    cache.update(torch.ones(2, 3, 3, 4), torch.ones(2, 3, 3, 4))
    original_clone = Tensor.clone
    clones: list[Tensor] = []

    def clone(
        tensor: Tensor,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> Tensor:
        clones.append(tensor)
        return original_clone(tensor, memory_format=memory_format)

    monkeypatch.setattr(Tensor, "clone", clone)
    # KVCache's contract uses equal batch and update-sequence widths here.
    keys = torch.full((2, 3, 2, 4), 2.0)
    values = torch.full((2, 3, 2, 4), -2.0)

    cache.update(keys, values)

    assert clones == []
    assert cache.length == cache.seen == 5
    torch.testing.assert_close(
        cache.k[0, 0, :, 0],
        torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0]),
    )
    torch.testing.assert_close(
        cache.v[0, 0, :, 0],
        torch.tensor([1.0, 1.0, 1.0, -2.0, -2.0]),
    )


def test_kv_cache_basic():
    cache = KVCache.alloc(batch=2, num_heads=4, max_seq=16, channels_head=8)
    assert cache.length == 0
    k = torch.randn(2, 4, 3, 8)
    v = torch.randn(2, 4, 3, 8)
    k_out, _v_out = cache.update(k, v)
    assert k_out.shape == (2, 4, 3, 8)
    assert cache.length == 3


def test_kv_cache_fifo_rolling():
    cache = KVCache.alloc(batch=2, num_heads=4, max_seq=4, channels_head=5)
    for i in range(4):
        k = torch.full((2, 4, 3, 5), float(i))
        v = torch.full((2, 4, 3, 5), float(i))
        cache.update(k, v)
    assert cache.length == 4
    # One more should FIFO.
    k = torch.full((2, 4, 3, 5), 99.0)
    v = torch.full((2, 4, 3, 5), 99.0)
    k_out, _v_out = cache.update(k, v)
    assert cache.length == 4
    assert k_out[0, 0, -1, 0].item() == 99.0
    assert k_out[0, 0, 0, 0].item() == 3.0


def test_kv_cache_freeze():
    cache = KVCache.alloc(batch=2, num_heads=3, max_seq=8, channels_head=5)
    k = torch.randn(2, 3, 4, 5)
    v = torch.randn(2, 3, 4, 5)
    cache.update(k, v)
    assert cache.length == 4

    frozen = cache.freeze()
    k2 = torch.randn(2, 3, 6, 5)
    v2 = torch.randn(2, 3, 6, 5)
    k_out, _v_out = frozen.update(k2, v2)
    assert frozen.length == 4
    assert k_out.shape == (2, 3, 4, 5)


def test_kv_cache_seen_tracks_absolute_position():
    """``seen`` counts total tokens ever written, even after FIFO eviction.

    Regression for MODEL-007: RoPE offset read ``length`` (capped at
    ``max_seq``), so absolute positions saturated/repeated once the
    cache filled. ``seen`` must keep counting past capacity.
    """
    cache = KVCache.alloc(batch=2, num_heads=4, max_seq=4, channels_head=5)
    seen: list[int] = []
    for i in range(4):
        seen.append(cache.seen)
        k = torch.full((2, 4, 3, 5), float(i))
        cache.update(k, k)
    assert seen == [0, 3, 6, 9]


def test_kv_cache_freeze_preserves_seen():
    """Freezing a past-capacity cache must keep the monotonic ``seen`` count.

    ``freeze()`` builds a snapshot; the constructor sets ``seen = length`` from
    the (FIFO-capped) length, dropping the true total so RoPE would reuse
    absolute positions after the freeze. The snapshot must carry ``seen``.
    """
    cache = KVCache.alloc(batch=2, num_heads=4, max_seq=4, channels_head=5)
    for i in range(4):
        cache.update(
            torch.full((2, 4, 3, 5), float(i)),
            torch.full((2, 4, 3, 5), float(i)),
        )
    assert cache.length == 4
    assert cache.seen == 4 * 3
    frozen = cache.freeze()
    assert frozen.seen == 4 * 3


def test_kv_cache_update_larger_than_capacity_raises():
    """A single update wider than ``max_seq`` is rejected, not silently sliced.

    Regression for MODEL-003: ``length + s - max_seq`` overflowing past
    ``length`` produced a negative ``keep`` and corrupt slices.
    """
    cache = KVCache.alloc(batch=1, num_heads=1, max_seq=2, channels_head=2)
    k = torch.randn(2, 3, 4, 5)
    with pytest.raises(ValueError, match="exceeds cache capacity"):
        cache.update(k, k)


def test_kv_cache_from_tensors_uses_sequence_axis_with_leading_dims() -> None:
    k = torch.randn(2, 3, 4, 5, 6)
    v = torch.randn(2, 3, 4, 5, 7)

    cache = KVCache(k, v)

    assert cache.k is k
    assert cache.v is v
    assert cache.length == 5
    assert cache.seen == 5


def test_kv_cache_update_uses_sequence_axis_with_multiple_leading_dims() -> None:
    cache = KVCache.alloc(
        batch=(2, 3),
        num_heads=4,
        max_seq=7,
        channels_head=6,
    )
    keys = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.float32).reshape(
        2,
        3,
        4,
        5,
        6,
    )
    values = -keys

    result_keys, result_values = cache.update(keys, values)

    assert cache.length == cache.seen == 5
    assert result_keys.shape == result_values.shape == (2, 3, 4, 5, 6)
    torch.testing.assert_close(result_keys, keys, rtol=0, atol=0)
    torch.testing.assert_close(result_values, values, rtol=0, atol=0)


def test_kv_cache_text(request: pytest.FixtureRequest) -> None:
    output = _cache_contract(torch.arange(120, dtype=torch.float32).reshape(2, 3, 5, 4))
    assert_text_golden(
        request,
        test_file=__file__,
        name="kv_cache",
        rendered=repr(output.tolist()),
    )


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_kv_cache_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="kv_cache",
        build_module=lambda: _Cache().to(device),
        build_input=lambda: torch.randn(2, 3, 5, 4),
        seed=0,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
