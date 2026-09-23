"""Tests for the pooling cost formulas."""

from __future__ import annotations

import pytest
import torch

from priml.model.pool import avg_pool_cost, max_pool_cost


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("kernel_size", [2, 3])
def test_max_pool_counts_values_the_saved_argmax_and_one_routed_gradient(
    dtype: torch.dtype,
    kernel_size: int,
) -> None:
    costed = max_pool_cost(channels=3, kernel_size=kernel_size, rows=5, dtype=dtype)
    window, itemsize = kernel_size**2, dtype.itemsize
    assert costed["bytes", "primal", "reduction", dtype] == itemsize * 15 * (window + 1)
    assert costed["bytes", "primal", "reduction", torch.int64] == 8 * 15
    assert costed["bytes", "adjoint", "selection", dtype] == itemsize * 15 * (
        window + 1
    )
    assert costed["bytes", "adjoint", "selection", torch.int64] == 8 * 15
    assert costed["flops", "primal", "reduction"].sum() == 15 * (window - 1)
    assert costed["flops", "adjoint", "selection"].sum() == 15
    assert costed.params == 0


def test_max_pool_accepts_a_per_axis_kernel() -> None:
    square = max_pool_cost(channels=2, kernel_size=3, rows=1, dtype=None)
    assert max_pool_cost(channels=2, kernel_size=(3, 3), rows=1, dtype=None) == square


def test_avg_pool_sums_each_channel_scales_once_and_spreads_back() -> None:
    costed = avg_pool_cost(channels=4, positions=9, batch_size=2, dtype=torch.float32)
    pooled = 4 * 2
    assert costed["flops", "primal", "reduction"].sum() == pooled * (9 - 1)
    assert costed["flops", "primal", "elementwise"].sum() == pooled
    assert costed["flops", "adjoint", "elementwise"].sum() == pooled * 9
    assert costed["bytes", "primal", "reduction"].sum() == 4 * (pooled * 9 + pooled)
    assert costed.params == 0


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
