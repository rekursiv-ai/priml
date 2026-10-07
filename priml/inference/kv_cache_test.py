"""Tests for the analytical KV-cache decode cost.

Shapes keep every axis distinct and above 1, and the query heads divide evenly
into key/value heads, so a swapped or dropped axis changes the answer.
"""

from __future__ import annotations

from typing import Final

from torch.utils.flop_counter import FlopCounterMode

import pytest
import torch

from priml.cost import peak
from priml.inference.kv_cache import KVCacheGeometry
from priml.testing.golden import assert_pprint_golden


NUM_LAYERS: Final = 3
NUM_HEADS: Final = 4
NUM_HEADS_KV: Final = 2
CHANNELS_HEAD: Final = 5
BF16: Final = torch.bfloat16


def kv_cache(
    *,
    num_heads_kv: int = NUM_HEADS_KV,
    dtype: torch.dtype = BF16,
) -> KVCacheGeometry:
    return KVCacheGeometry(
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_heads_kv=num_heads_kv,
        channels_head=CHANNELS_HEAD,
        dtype=dtype,
    )


def test_bytes_per_token_counts_keys_and_values_across_layers() -> None:
    assert (
        kv_cache().bytes_per_token()
        == 2 * NUM_LAYERS * NUM_HEADS_KV * CHANNELS_HEAD * BF16.itemsize
    )
    assert kv_cache(dtype=torch.int8).bytes_per_token() * BF16.itemsize == (
        kv_cache().bytes_per_token()
    )


def test_tokens_in_is_the_floor_of_the_budget_over_the_token_cost() -> None:
    cache = kv_cache()
    per_token = cache.bytes_per_token()
    assert cache.tokens_in(per_token * 5) == 5
    assert cache.tokens_in(per_token * 5 - 1) == 4
    assert cache.tokens_in(-1) == 0


@pytest.mark.parametrize(
    "cache",
    [
        KVCacheGeometry(),
        # Two unset sentinels multiply to a positive count.
        KVCacheGeometry(channels_head=CHANNELS_HEAD, num_heads_kv=NUM_HEADS_KV),
        KVCacheGeometry(
            num_layers=NUM_LAYERS,
            num_heads=3,
            num_heads_kv=2,
            channels_head=CHANNELS_HEAD,
        ),
    ],
    ids=["unset", "partly-set", "indivisible-heads"],
)
def test_an_invalid_geometry_raises_rather_than_pricing(cache: KVCacheGeometry) -> None:
    with pytest.raises(ValueError, match="num_"):
        cache.tokens_in(1 << 20)
    with pytest.raises(ValueError, match="num_"):
        cache.decode_cost(batch_size=2, context_len=8, device="h100", dtype=BF16)


@pytest.mark.parametrize("num_heads_kv", [NUM_HEADS, NUM_HEADS_KV])
def test_attention_flops_match_what_torch_counts_for_a_decode_step(
    num_heads_kv: int,
) -> None:
    batch_size, context_len = 3, 7
    group = NUM_HEADS // num_heads_kv
    # One query row: a decode step emits a single token.
    q = torch.zeros(batch_size, NUM_HEADS, 1, CHANNELS_HEAD)
    # The new token's keys join the cache before it attends, so T + 1 keys.
    k = torch.zeros(batch_size, num_heads_kv, context_len + 1, CHANNELS_HEAD)
    v = torch.zeros(batch_size, num_heads_kv, context_len + 1, CHANNELS_HEAD)
    with FlopCounterMode(display=False) as counter:
        scores = q @ k.repeat_interleave(group, dim=1).transpose(-1, -2)
        _ = scores.softmax(-1) @ v.repeat_interleave(group, dim=1)
    cost = kv_cache(num_heads_kv=num_heads_kv).decode_cost(
        batch_size=batch_size,
        context_len=context_len,
        device="h100",
        dtype=BF16,
    )
    assert cost.flops == NUM_LAYERS * counter.get_total_flops()


@pytest.mark.parametrize(
    ("num_heads_kv", "dtype"),
    [(NUM_HEADS, BF16), (NUM_HEADS_KV, BF16), (NUM_HEADS_KV, torch.int8)],
)
def test_cache_intensity_is_twice_the_grouping_ratio_over_itemsize(
    num_heads_kv: int,
    dtype: torch.dtype,
) -> None:
    cache = kv_cache(num_heads_kv=num_heads_kv, dtype=dtype)
    for context_len in (2, 8, 128):
        cost = cache.decode_cost(
            batch_size=2,
            context_len=context_len,
            device="h100",
            dtype=BF16,
        )
        assert cost.intensity == pytest.approx(
            2 * cache.grouping_ratio / dtype.itemsize,
        )


def test_the_weight_read_is_paid_once_and_its_arithmetic_scales_with_batch() -> None:
    num_params = 1 << 10
    one = kv_cache(dtype=torch.int8).decode_cost(
        batch_size=1,
        context_len=8,
        device="h100",
        dtype=BF16,
        num_params=num_params,
    )
    many = kv_cache(dtype=torch.int8).decode_cost(
        batch_size=4,
        context_len=8,
        device="h100",
        dtype=BF16,
        num_params=num_params,
    )
    # Weights are stored at the compute dtype, whatever the cache holds.
    assert one.weight_bytes == many.weight_bytes == num_params * BF16.itemsize
    assert many.kv_bytes == 4 * one.kv_bytes
    assert many.flops == 4 * one.flops


@pytest.mark.parametrize("device", ["rtx5050", "h100"])
def test_weight_bound_decode_turns_compute_bound_past_the_batch_ridge(
    device: str,
) -> None:
    ridge = peak()[device, BF16, "intensity", "matmul"]
    # Weights dominate the traffic at this context, so intensity is ~batch.
    crossover = ridge * BF16.itemsize / 2
    below, above = (
        kv_cache().decode_cost(
            batch_size=batch_size,
            context_len=2,
            device=device,
            dtype=BF16,
            num_params=1 << 30,
        )
        for batch_size in (int(crossover * 0.9), int(crossover * 1.1))
    )
    assert below.memory_bound
    assert not above.memory_bound


def test_the_ridge_is_read_at_the_compute_dtype_not_the_cache_dtype() -> None:
    cost = kv_cache(dtype=torch.float8_e4m3fn).decode_cost(
        batch_size=2,
        context_len=8,
        device="b200",
        dtype=BF16,
    )
    assert cost.ridge == peak()["b200", BF16, "intensity", "matmul"]
    assert cost.fraction_of_ridge == pytest.approx(cost.intensity / cost.ridge)


def test_a_dtype_the_device_cannot_run_raises_rather_than_reading_zero() -> None:
    with pytest.raises(ValueError, match="a100"):
        kv_cache().decode_cost(
            batch_size=2,
            context_len=8,
            device="a100",
            dtype=torch.float8_e4m3fn,
        )


@pytest.mark.parametrize(
    ("batch_size", "context_len", "num_params"),
    [(0, 8, 0), (2, -1, 0), (2, 8, -1)],
)
def test_an_invalid_step_raises_rather_than_pricing(
    batch_size: int,
    context_len: int,
    num_params: int,
) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        kv_cache().decode_cost(
            batch_size=batch_size,
            context_len=context_len,
            device="h100",
            dtype=BF16,
            num_params=num_params,
        )


def test_kv_cache_geometry_config_pprint() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="kv_cache_geometry",
        config=kv_cache(),
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
