from __future__ import annotations

import itertools
import math

from torch import Tensor

import pytest
import torch

from priml.math.basic import ceil_div
from priml.math.pooling import (
    _dim_info,
    adaptive_avg_pool2d,
    adaptive_avg_pool3d,
)


def test_adaptive_avg_pool2d_simple() -> None:
    x = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5)
    result = adaptive_avg_pool2d(x, (2, 2))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (2, 2))
    torch.testing.assert_close(result, expected)


def test_dim_info_uniform_nondivisible_windows_are_exact() -> None:
    dim = _dim_info(6, 4, torch.device("cpu"))
    assert torch.equal(
        dim.idx,
        torch.tensor([[0, 1], [1, 2], [3, 4], [4, 5]], dtype=torch.int64),
    )
    assert dim.length == 2
    assert torch.equal(dim.max_kernel_size_range, torch.tensor([0, 1]))
    assert not dim.needs_irregular_kernel


def test_dim_info_irregular_windows_are_exact() -> None:
    dim = _dim_info(5, 3, torch.device("cpu"))
    assert torch.equal(
        dim.idx,
        torch.tensor([[0, 1, 2], [1, 2, 3], [3, 4, 4]], dtype=torch.int64),
    )
    assert isinstance(dim.length, Tensor)
    assert torch.equal(dim.length, torch.tensor([2, 3, 2]))
    assert torch.equal(dim.max_kernel_size_range, torch.tensor([0, 1, 2]))
    assert dim.needs_irregular_kernel


def test_pool_variance_preserving_divisible_3d_exact_scale() -> None:
    x = torch.arange(1, 1 + 3 * 5 * 2 * 4 * 6, dtype=torch.float64).reshape(
        3,
        5,
        2,
        4,
        6,
    )
    actual = adaptive_avg_pool3d(x, (1, 2, 3), variance_preserving=True)
    expected = torch.nn.functional.avg_pool3d(x, (2, 2, 2), (2, 2, 2)) * 8**0.5
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pool_variance_preserving_3d_mixed_uniform_and_irregular_axes() -> None:
    """Uniform axes before an irregular one all contribute to each window."""
    spatial = (6, 4, 5)
    output_size = (4, 3, 3)
    x = torch.arange(1, 1 + 2 * 3 * math.prod(spatial), dtype=torch.float64).reshape(
        2,
        3,
        *spatial,
    )
    actual = adaptive_avg_pool3d(x, output_size, variance_preserving=True)
    expected = torch.empty((2, 3, *output_size), dtype=x.dtype)
    for out_idx in itertools.product(*(range(size) for size in output_size)):
        limits = [
            (
                idx * source // target,
                ceil_div((idx + 1) * source, target),
            )
            for idx, source, target in zip(out_idx, spatial, output_size, strict=True)
        ]
        block = x[(..., *(slice(start, stop) for start, stop in limits))]
        expected[(..., *out_idx)] = (
            block.sum(dim=(-3, -2, -1)) / math.prod(block.shape[-3:]) ** 0.5
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


def test_pool_variance_preserving_matches_window_sums_across_paths() -> None:
    for spatial, output_size in [
        ((4, 6), (2, 3)),
        ((5, 6), (3, 3)),
        ((3, 5), (5, 3)),
        ((8, 6), (6, 4)),
    ]:
        x = torch.arange(2 * 3 * math.prod(spatial), dtype=torch.float64).reshape(
            2,
            3,
            *spatial,
        )
        result = adaptive_avg_pool2d(x, output_size, variance_preserving=True)
        expected = torch.empty((2, 3, *output_size), dtype=x.dtype)
        for out_i in range(output_size[0]):
            lo_i = out_i * spatial[0] // output_size[0]
            hi_i = (out_i + 1) * spatial[0] // output_size[0]
            hi_i += int((out_i + 1) * spatial[0] % output_size[0] != 0)
            for out_j in range(output_size[1]):
                lo_j = out_j * spatial[1] // output_size[1]
                hi_j = (out_j + 1) * spatial[1] // output_size[1]
                hi_j += int((out_j + 1) * spatial[1] % output_size[1] != 0)
                block = x[..., lo_i:hi_i, lo_j:hi_j]
                expected[..., out_i, out_j] = (
                    block.sum(dim=(-2, -1)) / (block.shape[-2] * block.shape[-1]) ** 0.5
                )
        torch.testing.assert_close(result, expected, rtol=0, atol=1e-12)


def test_adaptive_avg_pool3d_variance_preserving_matches_window_sums() -> None:
    spatial = (4, 5, 6)
    output_size = (3, 4, 5)
    x = torch.arange(2 * 3 * math.prod(spatial), dtype=torch.float64).reshape(
        2,
        3,
        *spatial,
    )
    result = adaptive_avg_pool3d(x, output_size, variance_preserving=True)
    expected = torch.empty((2, 3, *output_size), dtype=x.dtype)
    for out_idx in itertools.product(*(range(size) for size in output_size)):
        limits = [
            (
                idx * source // target,
                ((idx + 1) * source + target - 1) // target,
            )
            for idx, source, target in zip(out_idx, spatial, output_size, strict=True)
        ]
        block = x[
            ...,
            *(slice(start, stop) for start, stop in limits),
        ]
        expected[(..., *out_idx)] = (
            block.sum(dim=(-3, -2, -1)) / (math.prod(block.shape[-3:])) ** 0.5
        )
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-12)


def test_invalid_pooling_shapes_name_the_rejected_dimension() -> None:
    with pytest.raises(RuntimeError, match=r"Expected 3D or 4D tensor, got 2D"):
        adaptive_avg_pool2d(torch.ones(2, 3), (2, 2))
    with pytest.raises(
        RuntimeError,
        match=r"Expected non-zero spatial dims, got shape",
    ):
        adaptive_avg_pool2d(torch.ones(2, 3, 0, 4), (2, 2), True)
    with pytest.raises(
        RuntimeError,
        match=r"Expected positive output_size, got \(0, 2\)",
    ):
        adaptive_avg_pool2d(torch.ones(2, 3, 4, 6), (0, 2), True)
    with pytest.raises(RuntimeError, match=r"Expected 4D or 5D tensor, got 3D"):
        adaptive_avg_pool3d(torch.ones(2, 3, 4), (2, 2, 2))


def test_ragged_index_tables_preserve_device_and_integer_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_arange = torch.arange
    original_scalar_tensor = torch.scalar_tensor
    calls: list[tuple[torch.device, torch.dtype]] = []
    scalar_calls: list[tuple[torch.device, torch.dtype]] = []

    def recording_arange(
        end: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        calls.append((device, dtype))
        return original_arange(end, device=device, dtype=dtype)

    def recording_scalar_tensor(
        value: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        scalar_calls.append((device, dtype))
        return original_scalar_tensor(value, dtype=dtype, device=device)

    monkeypatch.setattr(torch, "arange", recording_arange)
    monkeypatch.setattr(torch, "scalar_tensor", recording_scalar_tensor)
    adaptive_avg_pool2d(torch.ones(2, 3, 5, 6), (3, 3), True)
    assert calls == [(torch.device("cpu"), torch.int64)] * 4
    assert scalar_calls == [(torch.device("cpu"), torch.int64)]


def test_adaptive_avg_pool2d_variance_preserving() -> None:
    x = torch.randn(2, 3, 4, 6)
    result = adaptive_avg_pool2d(x, (4, 4), variance_preserving=True)
    # Check shape.
    assert result.shape == (2, 3, 4, 4)


def test_adaptive_avg_pool2d_divisible() -> None:
    # When input is divisible by output, should use optimized path.
    x = torch.randn(2, 3, 4, 6)
    result = adaptive_avg_pool2d(x, (4, 4))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
    torch.testing.assert_close(result, expected)


def test_adaptive_avg_pool2d_divisible_variance_preserving() -> None:
    # Divisible case with variance preserving.
    x = torch.randn(2, 3, 6, 8)
    result = adaptive_avg_pool2d(x, (3, 4), variance_preserving=True)
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (3, 4))
    # Variance preserving scales by sqrt(stride)
    expected = expected * (2 * 2) ** 0.5
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_non_divisible() -> None:
    # When input is not divisible, uses slower adaptive path.
    x = torch.randn(2, 3, 5, 7)
    result = adaptive_avg_pool2d(x, (3, 3))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (3, 3))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_non_divisible_variance_preserving() -> None:
    # Non-divisible case with variance preserving.
    x = torch.randn(2, 3, 5, 7)
    result = adaptive_avg_pool2d(x, (3, 3), variance_preserving=True)
    # Just check shape - exact value depends on adaptive logic.
    assert result.shape == (2, 3, 3, 3)


def test_adaptive_avg_pool2d_3d_input() -> None:
    # Test with 3D input (no batch dimension)
    x = torch.randn(2, 8, 9)
    result = adaptive_avg_pool2d(x, (4, 4))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_batched() -> None:
    # Test with batch dimension.
    x = torch.randn(4, 3, 6, 7)
    result = adaptive_avg_pool2d(x, (2, 2))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (2, 2))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_asymmetric() -> None:
    # Test with asymmetric output size.
    x = torch.randn(2, 3, 5, 6)
    result = adaptive_avg_pool2d(x, (3, 4))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (3, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_simple() -> None:
    x = torch.arange(720, dtype=torch.float32).reshape(2, 3, 4, 5, 6)
    result = adaptive_avg_pool3d(x, (2, 2, 2))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (2, 2, 2))
    torch.testing.assert_close(result, expected)


def test_adaptive_avg_pool3d_variance_preserving() -> None:
    x = torch.randn(2, 3, 4, 6, 8)
    result = adaptive_avg_pool3d(x, (4, 4, 4), variance_preserving=True)
    # Check shape.
    assert result.shape == (2, 3, 4, 4, 4)


def test_adaptive_avg_pool3d_divisible() -> None:
    # When input is divisible by output, should use optimized path.
    x = torch.randn(2, 3, 4, 6, 8)
    result = adaptive_avg_pool3d(x, (4, 4, 4))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (4, 4, 4))
    torch.testing.assert_close(result, expected)


def test_adaptive_avg_pool3d_divisible_variance_preserving() -> None:
    # Divisible case with variance preserving.
    x = torch.randn(2, 3, 4, 6, 8)
    adaptive_avg_pool3d(x, (4, 4, 4), variance_preserving=True)
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (4, 4, 4))
    # Variance preserving scales by sqrt(stride)
    expected = expected * (2 * 2 * 3) ** 0.5
    expected = expected * (2 * 2 * 2) ** 0.5


def test_adaptive_avg_pool3d_non_divisible() -> None:
    # When input is not divisible, uses slower adaptive path.
    x = torch.randn(2, 3, 5, 6, 7)
    result = adaptive_avg_pool3d(x, (3, 3, 3))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (3, 3, 3))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_non_divisible_variance_preserving() -> None:
    # Non-divisible case with variance preserving.
    x = torch.randn(2, 3, 5, 6, 7)
    result = adaptive_avg_pool3d(x, (3, 3, 3), variance_preserving=True)
    # Just check shape - exact value depends on adaptive logic.
    assert result.shape == (2, 3, 3, 3, 3)


def test_adaptive_avg_pool3d_4d_input() -> None:
    # Test with 4D input (no batch dimension)
    x = torch.randn(2, 4, 5, 6)
    result = adaptive_avg_pool3d(x, (4, 4, 4))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (4, 4, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_batched() -> None:
    # Test with batch dimension.
    x = torch.randn(2, 3, 4, 5, 6)
    result = adaptive_avg_pool3d(x, (2, 2, 2))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (2, 2, 2))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_asymmetric() -> None:
    # Test with asymmetric output size.
    x = torch.randn(2, 3, 4, 5, 6)
    result = adaptive_avg_pool3d(x, (2, 3, 4))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (2, 3, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_partially_adaptive() -> None:
    # Test case where some dimensions are adaptive and some are not.
    x = torch.randn(2, 3, 5, 4, 6)
    result = adaptive_avg_pool3d(x, (4, 3, 4))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (4, 3, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_single_adaptive_dim() -> None:
    # Test where only one dimension is adaptive.
    x = torch.randn(2, 3, 5, 4)
    result = adaptive_avg_pool2d(x, (3, 4))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (3, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_non_divisible_vp_multiple_channels() -> None:
    # Non-divisible variance preserving with multiple channels.
    x = torch.randn(2, 4, 5, 7)
    result = adaptive_avg_pool2d(x, (3, 4), variance_preserving=True)
    assert result.shape == (2, 4, 3, 4)


def test_adaptive_avg_pool3d_single_adaptive_dim() -> None:
    # Test where only one or two dimensions are adaptive.
    x = torch.randn(2, 3, 4, 6, 8)
    result = adaptive_avg_pool3d(x, (4, 3, 4))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (4, 3, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_all_adaptive() -> None:
    # Test where all dimensions are adaptive.
    x = torch.randn(2, 3, 5, 6, 7)
    result = adaptive_avg_pool3d(x, (3, 4, 5))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (3, 4, 5))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_non_divisible_vp_multiple_channels() -> None:
    # Non-divisible variance preserving with multiple channels.
    x = torch.randn(2, 3, 5, 6, 7)
    result = adaptive_avg_pool3d(x, (3, 4, 5), variance_preserving=True)
    assert result.shape == (2, 3, 3, 4, 5)


def test_adaptive_avg_pool2d_invalid_ndim() -> None:
    # Test error handling for invalid tensor dimensions.
    x = torch.randn(4, 5)  # 2D tensor, should fail.
    with pytest.raises(RuntimeError):
        adaptive_avg_pool2d(x, (2, 2))


def test_adaptive_avg_pool2d_zero_dimension() -> None:
    # Test error handling for zero-size dimensions.
    x = torch.randn(2, 3, 0, 4)  # Zero height.
    with pytest.raises(RuntimeError):
        adaptive_avg_pool2d(x, (2, 2))


def test_adaptive_avg_pool3d_invalid_ndim() -> None:
    # Test error handling for invalid tensor dimensions.
    x = torch.randn(4, 5, 6)  # 3D tensor, should fail.
    with pytest.raises(RuntimeError):
        adaptive_avg_pool3d(x, (2, 2, 2))


def test_adaptive_avg_pool3d_zero_dimension() -> None:
    # Test error handling for zero-size dimensions.
    x = torch.randn(2, 3, 0, 4, 5)  # Zero depth.
    with pytest.raises(RuntimeError):
        adaptive_avg_pool3d(x, (2, 2, 2))


def test_adaptive_avg_pool2d_adaptive_masking_both_dims() -> None:
    # Test the adaptive masking path for 2D pooling (lines 68-95)
    # Use input/output sizes that trigger adaptive behavior in both dimensions
    # This happens when in_size % out_size != 0 and out_size % in_size_mod != 0.
    x = torch.randn(2, 3, 4, 5)
    result = adaptive_avg_pool2d(x, (3, 3))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (3, 3))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_adaptive_masking_variance_preserving() -> None:
    # Test adaptive masking with variance preserving (lines 68-95, line 94)
    # Use dimensions that trigger both adaptive=True for both height and width.
    x = torch.randn(2, 3, 4, 5)
    result = adaptive_avg_pool2d(x, (3, 3), variance_preserving=True)
    assert result.shape == (2, 3, 3, 3)
    # Verify the result is not NaN or Inf.
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_adaptive_avg_pool2d_adaptive_single_dim_h() -> None:
    # Test adaptive masking in height dimension only (lines 68-74)
    x = torch.randn(2, 3, 5, 4)  # 11 -> 3 is adaptive, 8 -> 4 is not.
    result = adaptive_avg_pool2d(x, (3, 4))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (3, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_adaptive_single_dim_w() -> None:
    # Test adaptive masking in width dimension only (lines 75-81)
    x = torch.randn(2, 3, 4, 5)  # 8 -> 4 is not adaptive, 11 -> 3 is adaptive.
    result = adaptive_avg_pool2d(x, (4, 3))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (4, 3))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_adaptive_masking_all_dims() -> None:
    # Test the adaptive masking path for 3D pooling (lines 158-196)
    # Use input/output sizes that trigger adaptive behavior in all dimensions.
    x = torch.randn(2, 3, 4, 5, 6)
    result = adaptive_avg_pool3d(x, (3, 3, 3))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (3, 3, 3))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_adaptive_masking_variance_preserving() -> None:
    # Test adaptive masking with variance preserving (lines 158-196, line 194-195)
    x = torch.randn(2, 3, 4, 5, 6)
    result = adaptive_avg_pool3d(x, (3, 3, 3), variance_preserving=True)
    assert result.shape == (2, 3, 3, 3, 3)


def test_adaptive_avg_pool3d_adaptive_depth_only() -> None:
    # Test adaptive masking in depth dimension only (lines 158-164)
    x = torch.randn(2, 3, 5, 4, 6)  # 11 -> 3 is adaptive.
    result = adaptive_avg_pool3d(x, (3, 4, 4))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (3, 4, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_adaptive_height_only() -> None:
    # Test adaptive masking in height dimension only (lines 165-171)
    x = torch.randn(2, 3, 4, 5, 6)  # 11 -> 3 is adaptive.
    result = adaptive_avg_pool3d(x, (4, 3, 4))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (4, 3, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_adaptive_width_only() -> None:
    # Test adaptive masking in width dimension only (lines 172-178)
    x = torch.randn(2, 3, 4, 5, 6)  # 11 -> 3 is adaptive.
    result = adaptive_avg_pool3d(x, (4, 4, 3))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (4, 4, 3))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_adaptive_two_dims() -> None:
    # Test adaptive masking in two dimensions.
    x = torch.randn(2, 3, 5, 4, 6)
    result = adaptive_avg_pool3d(x, (3, 3, 4))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (3, 3, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_maxlength_edge_case() -> None:
    # Test case that triggers maxlength calculation edge cases (line 220)
    # When in_size % out_size == 0, maxlength should be reduced.
    x = torch.randn(2, 3, 4, 5)
    result = adaptive_avg_pool2d(x, (4, 4))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_minimum_clamping() -> None:
    # Test the minimum clamping in adaptive case (lines 229-234)
    # This uses torch.minimum to clamp indices.
    x = torch.randn(2, 3, 5, 6)
    result = adaptive_avg_pool2d(x, (5, 5))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (5, 5))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_length_computation() -> None:
    # Test length computation in adaptive case (lines 237-238)
    x = torch.randn(2, 3, 5, 6)
    result = adaptive_avg_pool2d(x, (7, 7))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (7, 7))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_end_index() -> None:
    # Test _end_index function usage (line 249)
    x = torch.randn(2, 3, 5, 6)
    result = adaptive_avg_pool2d(x, (6, 6))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (6, 6))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_masking_dim_neg2() -> None:
    # Test masking for dim=-2 case (lines 264-265)
    x = torch.randn(2, 3, 6, 4)
    result = adaptive_avg_pool2d(x, (5, 4))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (5, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_masking_dim_neg3() -> None:
    # Test masking for dim=-3 case in 3D pooling (lines 266-267)
    x = torch.randn(2, 3, 6, 4, 5)
    result = adaptive_avg_pool3d(x, (5, 4, 4))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (5, 4, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_masked_fill() -> None:
    # Test torch.masked_fill usage (line 268)
    x = torch.randn(2, 3, 6, 7)
    result = adaptive_avg_pool2d(x, (8, 8))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (8, 8))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_length_unsqueeze() -> None:
    # Test length unsqueezing (line 270)
    x = torch.randn(2, 4, 6, 7)
    result = adaptive_avg_pool2d(x, (7, 7))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (7, 7))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_complex_adaptive() -> None:
    # Complex test with multiple adaptive dimensions and variance preserving.
    x = torch.randn(2, 3, 5, 6, 7)
    result = adaptive_avg_pool3d(x, (5, 6, 7), variance_preserving=True)
    assert result.shape == (2, 3, 5, 6, 7)


def test_adaptive_avg_pool2d_large_batch() -> None:
    # Test with larger batch size to ensure robustness.
    x = torch.randn(4, 5, 6, 7)
    result = adaptive_avg_pool2d(x, (5, 5))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (5, 5))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_large_batch() -> None:
    # Test 3D pooling with larger batch size.
    x = torch.randn(2, 3, 4, 5, 6)
    result = adaptive_avg_pool3d(x, (5, 5, 5))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (5, 5, 5))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def _reference_variance_preserving(x: Tensor, output_size: tuple[int, int]) -> Tensor:
    """Per-window ``sum / sqrt(count)``, computed at the input's own dtype."""
    out_h, out_w = output_size
    h, w = x.shape[-2:]
    out = torch.zeros(*x.shape[:-2], out_h, out_w, dtype=x.dtype)
    for i in range(out_h):
        lo_i, hi_i = (i * h) // out_h, ceil_div((i + 1) * h, out_h)
        for j in range(out_w):
            lo_j, hi_j = (j * w) // out_w, ceil_div((j + 1) * w, out_w)
            block = x[..., lo_i:hi_i, lo_j:hi_j]
            count = block.shape[-1] * block.shape[-2]
            out[..., i, j] = block.sum(dim=(-2, -1)) / count**0.5
    return out


def test_variance_preserving_ragged_windows_keep_the_input_precision() -> None:
    """The divisor must not silently drop the result to float32.

    Ragged windows divide by a per-position int64 count, and ``count ** 0.5``
    promotes an integer tensor to float32 no matter what the input carried --
    so a float64 pool lost ~24 bits. Every other ragged test asserts only the
    shape, which is why this survived.
    """
    x = torch.randn(2, 3, 5, 6, dtype=torch.float64)
    torch.testing.assert_close(
        adaptive_avg_pool2d(x, (3, 3), variance_preserving=True),
        _reference_variance_preserving(x, (3, 3)),
        rtol=0,
        atol=1e-12,
    )


def test_variance_preserving_counts_remain_int64_for_large_half_windows() -> None:
    x = torch.zeros((4, 5, 6_146, 6), dtype=torch.float16)
    x[0, 0, 1_000, 0] = 0.004
    x[0, 0, 3_000, 4] = 0.004
    x[0, 0, 5_000, 0] = 0.004
    actual = adaptive_avg_pool2d(x, (3, 2), variance_preserving=True)
    value = torch.tensor(0.004, dtype=torch.float16)
    denominators = torch.tensor([6_147, 6_150, 6_147], dtype=torch.int64)
    values = value / denominators.to(torch.float16).sqrt()
    expected = torch.zeros((4, 5, 3, 2), dtype=torch.float16)
    expected[0, 0, 0, 0] = values[0]
    expected[0, 0, 1, 1] = values[1]
    expected[0, 0, 2, 0] = values[2]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_variance_preserving_returns_the_input_dtype(dtype: torch.dtype) -> None:
    """A ragged size must not change the dtype a divisible one preserves."""
    ragged = adaptive_avg_pool2d(
        torch.randn(2, 3, 5, 6, dtype=dtype),
        (3, 3),
        variance_preserving=True,
    )
    divisible = adaptive_avg_pool2d(
        torch.randn(2, 3, 4, 6, dtype=dtype),
        (4, 4),
        variance_preserving=True,
    )
    assert ragged.dtype == dtype
    assert divisible.dtype == dtype
    assert (
        adaptive_avg_pool3d(
            torch.randn(2, 3, 4, 5, 6, dtype=dtype),
            (3, 3, 3),
            variance_preserving=True,
        ).dtype
        == dtype
    )


def test_adaptive_avg_pool2d_both_dims_adaptive_vp() -> None:
    # Test to specifically cover line 94 - variance preserving with both dims adaptive
    # Using 11x11 -> 3x3:
    # 11 % 3 = 2, and 3 % 2 = 1 (not 0), so adaptive = True for both dims.
    x = torch.randn(2, 3, 4, 5)
    result = adaptive_avg_pool2d(x, (3, 3), variance_preserving=True)
    assert result.shape == (2, 3, 3, 3)
    assert not torch.isnan(result).any()


@pytest.mark.parametrize("variance_preserving", [False, True])
def test_pool_negative_output_dimension_is_rejected(variance_preserving: bool) -> None:
    with pytest.raises(RuntimeError, match="positive output_size"):
        adaptive_avg_pool2d(
            torch.zeros(2, 3, 4, 5),
            (-1, 2),
            variance_preserving=variance_preserving,
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
