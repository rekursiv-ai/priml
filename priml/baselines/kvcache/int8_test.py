"""Tests for the int8 KV cache exp001 injects into the decode chain."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Final

from torch import Tensor

import pytest
import torch

from priml.baselines.kvcache.decode import _cache_bytes, decoder
from priml.baselines.kvcache.decode_test import (
    CHANNELS_HEAD,
    NUM_DECODE_STEPS,
    NUM_HEADS_KV,
    NUM_LAYERS,
    PROMPT_LEN,
    VOCAB,
    _prompt,
    _tiny,
)
from priml.baselines.kvcache.int8 import (
    SCALE_COLUMNS,
    Int8KVCache,
    int8_layer_cache,
)
from priml.model.attention.kvcache import KVCache, alloc_layer_cache
from priml.model.attention.mla import MultiHeadLatentAttention


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.model.transformer.transformer import Transformer


_NUM_HEADS: Final = 3
_CHANNELS_HEAD_CACHE: Final = 4
_CODE_MAX: Final = 127.0


def _small_decoder() -> Transformer.Config:
    """Return ``decoder()`` at two layers; its 64-wide cache rows stay."""
    cfg = decoder()
    cfg.channels_in, cfg.num_layers = 128, 2
    return cfg


def test_int8_cache_counts_its_scales_inside_the_cache_tensors() -> None:
    model = _small_decoder().make()
    max_seq = 64
    default = alloc_layer_cache(
        model,
        batch=2,
        max_seq=max_seq,
        dtype=torch.float32,
    )
    packed = int8_layer_cache(
        model,
        batch=2,
        max_seq=max_seq,
        dtype=torch.float32,
    )
    float_bytes = _cache_bytes(default)
    int8_bytes = _cache_bytes(packed)

    first = next(iter(default.values()))
    assert isinstance(first, KVCache)
    channels = first.k.shape[-1]
    rows = len(default) * first.k.shape[0] * first.k.shape[1] * max_seq
    # exp000's storage at the model's dtype, spelled out so the identity
    # below does not depend on the model's dtype: two values, two bytes.
    bf16_bytes = rows * 2 * channels * torch.bfloat16.itemsize
    # Codes replace the two-byte halves; the float32 scales ride in-row and
    # are counted too, which is what keeps this from being half exactly.
    assert int8_bytes == bf16_bytes // 2 + rows * 2 * SCALE_COLUMNS
    assert int8_bytes < bf16_bytes
    assert int8_bytes < float_bytes


def test_update_stays_within_half_a_quantization_step() -> None:
    cache = _cache()
    k = torch.randn(2, _NUM_HEADS, 5, _CHANNELS_HEAD_CACHE) * 3.0
    v = torch.randn(2, _NUM_HEADS, 5, _CHANNELS_HEAD_CACHE) * 0.5

    k_all, v_all = cache.update(k, v)

    scale_k = k.abs().amax(dim=-1, keepdim=True) / _CODE_MAX
    scale_v = v.abs().amax(dim=-1, keepdim=True) / _CODE_MAX
    # Rounding costs at most half a step; the 0.51 leaves headroom for the
    # float32 multiply that reads the codes back.
    assert torch.all((k_all - k).abs() <= scale_k * 0.51)
    assert torch.all((v_all - v).abs() <= scale_v * 0.51)
    assert k_all.shape == k.shape
    assert cache.length == 5
    assert cache.seen == 5


def test_update_answers_in_the_incoming_dtype() -> None:
    cache = _cache()
    k = torch.randn(2, _NUM_HEADS, 5, _CHANNELS_HEAD_CACHE, dtype=torch.bfloat16)
    v = torch.randn(2, _NUM_HEADS, 5, _CHANNELS_HEAD_CACHE, dtype=torch.bfloat16)

    k_all, v_all = cache.update(k, v)

    assert k_all.dtype is torch.bfloat16
    assert v_all.dtype is torch.bfloat16


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("length", [0, 1, 2])
def test_scale_unpacking_handles_every_row_alignment(
    channels: int,
    length: int,
) -> None:
    # Singleton axes expose _dequantized's invalid scale strides under plain clone.
    cache = Int8KVCache(
        k=torch.zeros(1, 1, 2, channels + SCALE_COLUMNS, dtype=torch.int8),
        v=torch.zeros(1, 1, 2, channels + SCALE_COLUMNS, dtype=torch.int8),
        length=0,
    )
    # Length 0/1 exposes _dequantized's empty/singleton dtype-view stride failure.
    incoming = torch.ones(1, 1, length, channels)

    k_all, v_all = cache.update(incoming, incoming)

    torch.testing.assert_close(k_all, incoming, rtol=0, atol=0)
    torch.testing.assert_close(v_all, incoming, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("dtype", "value"),
    [(torch.float16, 1e-6), (torch.bfloat16, 2e-38), (torch.float64, 1e39)],
)
def test_dequantization_preserves_the_inputs_dynamic_range(
    dtype: torch.dtype,
    value: float,
) -> None:
    cache = _cache()
    incoming = torch.full(
        (2, _NUM_HEADS, 5, _CHANNELS_HEAD_CACHE),
        fill_value=value,
        dtype=dtype,
    )

    k_all, v_all = cache.update(incoming, incoming)

    torch.testing.assert_close(k_all, incoming, rtol=1 / 127, atol=0)
    torch.testing.assert_close(v_all, incoming, rtol=1 / 127, atol=0)


def test_channel_broadcasting_matches_the_float_cache_after_eviction() -> None:
    cache = _cache(max_seq=6)
    control = KVCache.alloc(
        batch=2,
        num_heads=_NUM_HEADS,
        max_seq=6,
        channels_head=_CHANNELS_HEAD_CACHE,
    )
    initial = torch.ones(2, _NUM_HEADS, 6, _CHANNELS_HEAD_CACHE)
    keys = torch.full((2, _NUM_HEADS, 5, _CHANNELS_HEAD_CACHE), fill_value=2.0)
    # A singleton channel exercises Int8KVCache.update's expansion after eviction.
    values = torch.full((2, _NUM_HEADS, 5, 1), fill_value=3.0)
    cache.update(initial, initial)
    control.update(initial, initial)

    actual = cache.update(keys, values)
    expected = control.update(keys, values)

    for actual_half, expected_half in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_half, expected_half, rtol=0, atol=0)
    assert (cache.length, cache.seen) == (control.length, control.seen)


@pytest.mark.parametrize("batch", [2, (2, 4)])
def test_alloc_builds_packed_storage_with_independent_key_and_value_widths(
    batch: int | tuple[int, ...],
) -> None:
    cache = Int8KVCache.alloc(
        batch=batch,
        num_heads=_NUM_HEADS,
        max_seq=5,
        channels_head=6,
        channels_v_head=7,
        dtype=torch.float64,
    )
    batch_shape = (batch,) if isinstance(batch, int) else batch
    assert isinstance(cache, Int8KVCache)
    assert cache.k.dtype is torch.int8
    assert cache.v.dtype is torch.int8
    assert cache.k.shape == (*batch_shape, _NUM_HEADS, 5, 6 + SCALE_COLUMNS)
    assert cache.v.shape == (*batch_shape, _NUM_HEADS, 5, 7 + SCALE_COLUMNS)
    assert (cache.length, cache.seen) == (0, 0)
    keys = torch.ones(*batch_shape, _NUM_HEADS, 5, 6)
    values = torch.ones(*batch_shape, _NUM_HEADS, 5, 7)

    k_all, v_all = cache.update(keys, values)

    torch.testing.assert_close(k_all, keys, rtol=0, atol=0)
    torch.testing.assert_close(v_all, values, rtol=0, atol=0)


def test_quantization_keeps_nan_visible_to_attention() -> None:
    cache = _cache()
    incoming = torch.ones(2, _NUM_HEADS, 5, _CHANNELS_HEAD_CACHE)
    incoming[..., 0] = torch.nan

    k_all, v_all = cache.update(incoming, incoming)

    assert torch.isnan(k_all).any(dim=-1).all()
    assert torch.isnan(v_all).any(dim=-1).all()


def test_fifo_shift_keeps_codes_and_scales_aligned() -> None:
    cache = _cache(max_seq=6)
    first = torch.randn(2, _NUM_HEADS, 6, _CHANNELS_HEAD_CACHE)
    cache.update(first, first)
    more = torch.randn(2, _NUM_HEADS, 5, _CHANNELS_HEAD_CACHE)

    k_all, _ = cache.update(more, more)

    expected = torch.cat((first[..., more.shape[-2] :, :], more), dim=-2)
    scale = expected.abs().amax(dim=-1, keepdim=True) / _CODE_MAX
    assert torch.all((k_all - expected).abs() <= scale * 0.51)
    assert cache.length == first.shape[-2]
    assert cache.seen == first.shape[-2] + more.shape[-2]


def test_progress_matches_the_float_cache_on_the_same_writes() -> None:
    control = KVCache.alloc(
        batch=2,
        num_heads=_NUM_HEADS,
        max_seq=6,
        channels_head=_CHANNELS_HEAD_CACHE,
        dtype=torch.float32,
    )
    cache = _cache(max_seq=6)
    writes = [
        torch.randn(2, _NUM_HEADS, 5, _CHANNELS_HEAD_CACHE),
        torch.randn(2, _NUM_HEADS, 5, _CHANNELS_HEAD_CACHE),
        torch.randn(2, _NUM_HEADS, 6, _CHANNELS_HEAD_CACHE),
    ]
    for incoming in writes:
        control.update(incoming, incoming)
        cache.update(incoming, incoming)

    assert cache.length == control.length
    assert cache.seen == control.seen
    assert cache.max_seq == control.max_seq


def test_a_zero_row_dequantizes_to_zero() -> None:
    cache = _cache()
    zeros = torch.zeros(2, _NUM_HEADS, 5, _CHANNELS_HEAD_CACHE)

    k_all, v_all = cache.update(zeros, zeros)

    assert torch.isfinite(k_all).all()
    assert torch.isfinite(v_all).all()
    assert torch.equal(k_all, zeros)
    assert torch.equal(v_all, zeros)


def test_int8_layer_cache_widens_every_slot_the_model_allocates() -> None:
    model = _small_decoder().make()
    default = alloc_layer_cache(
        model,
        batch=2,
        max_seq=16,
        dtype=torch.float32,
    )
    packed = int8_layer_cache(
        model,
        batch=2,
        max_seq=16,
        dtype=torch.float32,
    )

    assert packed.keys() == default.keys()
    for key, state in packed.items():
        assert isinstance(state, Int8KVCache)
        base = default[key]
        assert isinstance(base, KVCache)
        assert state.k.dtype is torch.int8
        assert state.v.dtype is torch.int8
        assert state.k.shape == (
            *base.k.shape[:-1],
            base.k.shape[-1] + SCALE_COLUMNS,
        )
        assert state.v.shape == (
            *base.v.shape[:-1],
            base.v.shape[-1] + SCALE_COLUMNS,
        )
        assert state.length == 0
        assert state.seen == 0


@pytest.mark.parametrize("model_device", ["cpu", "meta"])
@pytest.mark.parametrize("cache_device", ["cpu", "meta"])
def test_layer_allocator_honors_the_requested_device(
    model_device: str,
    cache_device: str,
) -> None:
    model = _tiny().model.make().to(device=model_device)

    caches = int8_layer_cache(model, batch=2, max_seq=4, device=cache_device)

    for state in caches.values():
        assert isinstance(state, Int8KVCache)
        assert state.k.device == torch.device(cache_device)
        assert state.v.device == torch.device(cache_device)


def test_layer_allocator_allocates_only_the_packed_cache_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny().model.make()
    allocations: list[int] = []
    monkeypatch.setattr(
        torch,
        "zeros",
        partial(_record_allocation, make=torch.zeros, allocations=allocations),
    )

    caches = int8_layer_cache(model, batch=2, max_seq=4)

    assert sum(allocations) == _cache_bytes(caches)


def test_layer_allocator_preserves_asymmetric_latent_cache_widths() -> None:
    cfg = MultiHeadLatentAttention.Config()
    cfg.channels_in = 8
    cfg.num_heads = 2
    cfg.channels_qk_nope_head = 4
    cfg.channels_qk_rope_head = 2
    cfg.channels_v_head = 4
    cfg.kv_lora_rank = 3
    cfg.depth_index = ((0, 1),)
    model = cfg.make()

    caches = int8_layer_cache(model, batch=4, max_seq=5)
    state = caches[cfg.depth_index]
    assert isinstance(state, Int8KVCache)
    # MultiHeadLatentAttention.alloc_kv_cache fixes one head to avoid latent copies.
    assert state.k.shape == (4, 1, 5, cfg.kv_lora_rank + SCALE_COLUMNS)
    assert state.v.shape == (4, 1, 5, cfg.channels_qk_rope_head + SCALE_COLUMNS)
    output = model(torch.ones(4, 5, cfg.channels_in), cache=caches)

    assert output.shape == (4, 5, cfg.channels_in)
    assert torch.isfinite(output).all()
    assert state.length == 5


def test_layer_allocator_matches_the_float_allocator_without_a_rotary_key() -> None:
    cfg = MultiHeadLatentAttention.Config()
    cfg.channels_in = 8
    cfg.num_heads = 2
    cfg.channels_qk_nope_head = 4
    cfg.channels_qk_rope_head = 0
    cfg.channels_v_head = 4
    cfg.kv_lora_rank = 3
    cfg.depth_index = ((0, 1),)
    model = cfg.make()
    x = torch.ones(4, 5, cfg.channels_in)
    floats = alloc_layer_cache(model, batch=4, max_seq=5)
    packed = int8_layer_cache(model, batch=4, max_seq=5)

    expected = model(x, cache=floats)
    actual = model(x, cache=packed)

    float_state = floats[cfg.depth_index]
    state = packed[cfg.depth_index]
    assert isinstance(float_state, KVCache)
    assert isinstance(state, Int8KVCache)
    assert state.v.shape == (4, 1, 5, SCALE_COLUMNS)
    assert actual.shape == expected.shape
    assert state.length == float_state.length == 5


@pytest.mark.parametrize(
    ("channels_head", "channels_v_head"),
    [(-1, 2), (2, -1), (-1, None)],
)
def test_alloc_rejects_a_negative_width_as_the_float_cache_does(
    channels_head: int,
    channels_v_head: int | None,
) -> None:
    with pytest.raises(RuntimeError, match="non-negative"):
        KVCache.alloc(
            batch=2,
            num_heads=_NUM_HEADS,
            max_seq=5,
            channels_head=channels_head,
            channels_v_head=channels_v_head,
        )

    with pytest.raises(ValueError, match="non-negative"):
        Int8KVCache.alloc(
            batch=2,
            num_heads=_NUM_HEADS,
            max_seq=5,
            channels_head=channels_head,
            channels_v_head=channels_v_head,
        )


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_update_quantizes_float8_inputs_and_answers_in_float8(
    dtype: torch.dtype,
) -> None:
    incoming = torch.randn(2, _NUM_HEADS, 5, _CHANNELS_HEAD_CACHE).to(dtype)
    wide = incoming.to(torch.float32)

    k_all, v_all = _cache().update(incoming, incoming)
    expected, _ = _cache().update(wide, wide)

    assert k_all.dtype is v_all.dtype is dtype
    torch.testing.assert_close(
        k_all.float(),
        expected.to(dtype).float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        v_all.float(),
        expected.to(dtype).float(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("failure", ["length", "capacity", "shape", "copy"])
def test_rejected_updates_preserve_storage_and_progress(failure: str) -> None:
    cache = _cache(max_seq=6)
    initial = torch.ones(2, _NUM_HEADS, 6, _CHANNELS_HEAD_CACHE)
    cache.update(initial, initial)
    keys_before = cache.k.clone()
    values_before = cache.v.clone()
    keys = torch.ones(2, _NUM_HEADS, 5, _CHANNELS_HEAD_CACHE)
    values = keys
    if failure == "length":
        values = initial
    elif failure == "capacity":
        keys = values = torch.ones(2, _NUM_HEADS, 7, _CHANNELS_HEAD_CACHE)
    elif failure == "copy":
        # Right shape, but the values cannot be read: the keys would write first.
        values = torch.empty_like(keys, device="meta")
    else:
        # The head mismatch must fail before Int8KVCache.update evicts stored rows.
        values = torch.ones(2, _NUM_HEADS + 1, 5, _CHANNELS_HEAD_CACHE)

    with pytest.raises(
        (ValueError, NotImplementedError),
        match=r"keys and|capacity|Cannot write|meta",
    ):
        cache.update(keys, values)

    torch.testing.assert_close(cache.k, keys_before, rtol=0, atol=0)
    torch.testing.assert_close(cache.v, values_before, rtol=0, atol=0)
    assert (cache.length, cache.seen) == (initial.shape[-2], initial.shape[-2])


def test_update_refuses_a_width_the_cache_cannot_hold() -> None:
    cache = _cache()
    too_wide = torch.zeros(
        2,
        _NUM_HEADS,
        5,
        _CHANNELS_HEAD_CACHE + SCALE_COLUMNS + 1,
    )

    with pytest.raises(ValueError, match="Cannot write"):
        cache.update(too_wide, too_wide)


def test_freeze_is_refused_because_the_frozen_view_serves_raw_codes() -> None:
    with pytest.raises(NotImplementedError, match="freeze"):
        _cache().freeze()


def test_exp001_decodes_on_cpu_with_a_smaller_cache() -> None:
    cfg = _tiny()
    cfg.allocate_cache = int8_layer_cache
    first = cfg.make().measure()
    again = cfg.make().measure()
    control = _tiny().make().measure()

    assert torch.equal(first.tokens, again.tokens)
    assert first.tokens.shape == (len(first.tokens), PROMPT_LEN + NUM_DECODE_STEPS + 1)
    assert int(first.tokens.min()) >= 0
    assert int(first.tokens.max()) < VOCAB
    assert first.kv_bytes < control.kv_bytes
    assert first.kv_bytes_per_token == (
        NUM_LAYERS * 2 * NUM_HEADS_KV * (CHANNELS_HEAD + SCALE_COLUMNS)
    )


def test_the_prompt_the_end_to_end_run_feeds_is_still_exp000s() -> None:
    cfg = _tiny()
    cfg.allocate_cache = int8_layer_cache
    report = cfg.make().measure()

    assert torch.equal(report.tokens[:, :PROMPT_LEN], _prompt())


def _cache(*, batch: int = 2, max_seq: int = 5) -> Int8KVCache:
    """Build a tiny int8 cache directly from the documented layout."""
    return Int8KVCache(
        k=torch.zeros(
            (batch, _NUM_HEADS, max_seq, _CHANNELS_HEAD_CACHE + SCALE_COLUMNS),
            dtype=torch.int8,
        ),
        v=torch.zeros(
            (batch, _NUM_HEADS, max_seq, _CHANNELS_HEAD_CACHE + SCALE_COLUMNS),
            dtype=torch.int8,
        ),
        length=0,
    )


def _record_allocation(
    shape: tuple[int, ...],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    make: Callable[..., Tensor],
    allocations: list[int],
) -> Tensor:
    tensor = make(shape, device=device, dtype=dtype)
    allocations.append(tensor.nbytes)
    return tensor


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
