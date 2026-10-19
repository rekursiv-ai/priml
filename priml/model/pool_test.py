"""Tests for the pooling cost formulas."""

from __future__ import annotations

import pytest
import torch

from priml.model.pool import AvgPool2d, avg_pool_cost, max_pool_cost


@pytest.mark.parametrize("ceil_mode", [False, True])
@pytest.mark.parametrize("count_include_pad", [False, True])
@pytest.mark.parametrize("divisor_override", [None, 7])
def test_average_pool_matches_torch_complete_values_and_gradients(
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: int | None,
) -> None:
    config = AvgPool2d.Config()
    config.kernel_size = (3, 2)
    config.stride = (2, 3)
    config.padding = (1, 1)
    config.ceil_mode = ceil_mode
    config.count_include_pad = count_include_pad
    config.divisor_override = divisor_override
    native = config.make()
    reference = torch.nn.AvgPool2d(
        (3, 2),
        (2, 3),
        (1, 1),
        ceil_mode,
        count_include_pad,
        divisor_override,
    )
    value = torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 3, 4, 5) / 31
    left, right = value.clone().requires_grad_(), value.clone().requires_grad_()
    actual, expected = native(left, ignored=True), reference(right)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    assert torch.equal(
        torch.autograd.grad(actual.square().sum(), left)[0],
        torch.autograd.grad(expected.square().sum(), right)[0],
    )
    assert config.output_grid((4, 5)) == actual.shape[-2:]
    native.reset_parameters()
    assert native.state_dict() == {}


def test_pool_geometry_drops_a_ceil_window_starting_in_right_padding() -> None:
    config = AvgPool2d.Config()
    config.kernel_size = 2
    config.stride = 3
    config.padding = 1
    config.ceil_mode = True
    assert config.output_grid((3, 4)) == (2, 2)
    assert config.output_grid((4, 5)) == (2, 2)


def test_pool_cost_counts_clipped_visits_overlap_and_uncovered_inputs() -> None:
    config = AvgPool2d.Config()
    config.kernel_size = (3, 2)
    config.stride = (2, 3)
    config.padding = (1, 1)
    config.ceil_mode = True
    # Three row windows read 2+3+2 source rows; columns read 1+2+1 pixels.
    # The geometry is checked independently against torch in the tests above.
    rows, columns = config.output_grid((5, 6))
    visits = sum(
        max(0, min(5, r * 2 - 1 + 3) - max(0, r * 2 - 1)) for r in range(rows)
    ) * sum(max(0, min(6, c * 3 - 1 + 2) - max(0, c * 3 - 1)) for c in range(columns))
    covered = len(
        {r * 2 - 1 + k for r in range(rows) for k in range(3)} & set(range(5)),
    ) * len({c * 3 - 1 + k for c in range(columns) for k in range(2)} & set(range(6)))
    result = config.cost(
        input_grid=(5, 6),
        channels=3,
        batch_size=2,
        dtype=torch.float64,
    )
    config.channels_out = 3
    configured = config.copy_tree().finalize()
    assert configured.channels_in == configured.channels_out == 3
    assert (
        configured.cost(input_grid=(5, 6), batch_size=2, dtype=torch.float64) == result
    )
    scale, outputs, inputs = 6, 6 * rows * columns, 6 * 5 * 6
    assert result["flops", "primal", "reduction"].sum() == scale * visits - outputs
    assert result["flops", "primal", "elementwise"].sum() == outputs
    assert result["flops", "adjoint", "elementwise"].sum() == outputs
    assert result["flops", "adjoint", "reduction"].sum() == scale * (visits - covered)
    assert result["bytes", "adjoint", "reduction"].sum() == 8 * (
        scale * visits + inputs
    )


def test_average_pool_default_stride_and_invalid_rank() -> None:
    config = AvgPool2d.Config()
    config.kernel_size = 2
    assert config.output_grid((5, 7)) == (2, 3)
    with pytest.raises(ValueError, match="two spatial"):
        config.output_grid((5,))


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


def test_avg_pool_cost_tracks_each_cell_in_its_requested_dtype() -> None:
    dtype = torch.float64
    costed = avg_pool_cost(channels=3, positions=5, batch_size=2, dtype=dtype)
    assert costed["flops", "primal", "reduction", dtype] == 24
    assert costed["bytes", "primal", "reduction", dtype] == 288
    assert costed["flops", "primal", "elementwise", dtype] == 6
    assert costed["bytes", "primal", "elementwise", dtype] == 96
    assert costed["flops", "adjoint", "elementwise", dtype] == 30
    assert costed["bytes", "adjoint", "elementwise", dtype] == 288
    assert costed.params == costed.params_active == costed.bytes_state == 0


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
    from priml.lib.testing.runner import test_main

    test_main(__file__)
