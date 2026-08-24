from __future__ import annotations

import math

from torch import Tensor

import pytest
import torch

from priml.math.basic import ceil_div
from priml.math.pooling import adaptive_avg_pool2d, adaptive_avg_pool3d


def test_adaptive_avg_pool2d_simple() -> None:
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    result = adaptive_avg_pool2d(x, (2, 2))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (2, 2))
    torch.testing.assert_close(result, expected)


def test_adaptive_avg_pool2d_variance_preserving() -> None:
    x = torch.randn(2, 3, 8, 8)
    result = adaptive_avg_pool2d(x, (4, 4), variance_preserving=True)
    # Check shape
    assert result.shape == (2, 3, 4, 4)


def test_adaptive_avg_pool2d_divisible() -> None:
    # When input is divisible by output, should use optimized path
    x = torch.randn(1, 1, 8, 8)
    result = adaptive_avg_pool2d(x, (4, 4))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
    torch.testing.assert_close(result, expected)


def test_adaptive_avg_pool2d_divisible_variance_preserving() -> None:
    # Divisible case with variance preserving
    x = torch.randn(1, 2, 8, 8)
    result = adaptive_avg_pool2d(x, (4, 4), variance_preserving=True)
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
    # Variance preserving scales by sqrt(stride)
    expected = expected * (2 * 2) ** 0.5
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_non_divisible() -> None:
    # When input is not divisible, uses slower adaptive path
    x = torch.randn(1, 1, 7, 7)
    result = adaptive_avg_pool2d(x, (3, 3))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (3, 3))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_non_divisible_variance_preserving() -> None:
    # Non-divisible case with variance preserving
    x = torch.randn(1, 2, 7, 7)
    result = adaptive_avg_pool2d(x, (3, 3), variance_preserving=True)
    # Just check shape - exact value depends on adaptive logic
    assert result.shape == (1, 2, 3, 3)


def test_adaptive_avg_pool2d_3d_input() -> None:
    # Test with 3D input (no batch dimension)
    x = torch.randn(2, 8, 8)
    result = adaptive_avg_pool2d(x, (4, 4))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_batched() -> None:
    # Test with batch dimension
    x = torch.randn(4, 3, 6, 6)
    result = adaptive_avg_pool2d(x, (2, 2))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (2, 2))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_asymmetric() -> None:
    # Test with asymmetric output size
    x = torch.randn(1, 1, 9, 12)
    result = adaptive_avg_pool2d(x, (3, 4))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (3, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_simple() -> None:
    x = torch.arange(64, dtype=torch.float32).reshape(1, 1, 4, 4, 4)
    result = adaptive_avg_pool3d(x, (2, 2, 2))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (2, 2, 2))
    torch.testing.assert_close(result, expected)


def test_adaptive_avg_pool3d_variance_preserving() -> None:
    x = torch.randn(2, 3, 8, 8, 8)
    result = adaptive_avg_pool3d(x, (4, 4, 4), variance_preserving=True)
    # Check shape
    assert result.shape == (2, 3, 4, 4, 4)


def test_adaptive_avg_pool3d_divisible() -> None:
    # When input is divisible by output, should use optimized path
    x = torch.randn(1, 1, 8, 8, 8)
    result = adaptive_avg_pool3d(x, (4, 4, 4))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (4, 4, 4))
    torch.testing.assert_close(result, expected)


def test_adaptive_avg_pool3d_divisible_variance_preserving() -> None:
    # Divisible case with variance preserving
    x = torch.randn(1, 2, 8, 8, 8)
    result = adaptive_avg_pool3d(x, (4, 4, 4), variance_preserving=True)
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (4, 4, 4))
    # Variance preserving scales by sqrt(stride)
    expected = expected * (2 * 2 * 2) ** 0.5
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_non_divisible() -> None:
    # When input is not divisible, uses slower adaptive path
    x = torch.randn(1, 1, 7, 7, 7)
    result = adaptive_avg_pool3d(x, (3, 3, 3))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (3, 3, 3))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_non_divisible_variance_preserving() -> None:
    # Non-divisible case with variance preserving
    x = torch.randn(1, 2, 7, 7, 7)
    result = adaptive_avg_pool3d(x, (3, 3, 3), variance_preserving=True)
    # Just check shape - exact value depends on adaptive logic
    assert result.shape == (1, 2, 3, 3, 3)


def test_adaptive_avg_pool3d_4d_input() -> None:
    # Test with 4D input (no batch dimension)
    x = torch.randn(2, 8, 8, 8)
    result = adaptive_avg_pool3d(x, (4, 4, 4))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (4, 4, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_batched() -> None:
    # Test with batch dimension
    x = torch.randn(2, 3, 6, 6, 6)
    result = adaptive_avg_pool3d(x, (2, 2, 2))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (2, 2, 2))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_asymmetric() -> None:
    # Test with asymmetric output size
    x = torch.randn(1, 1, 6, 9, 12)
    result = adaptive_avg_pool3d(x, (2, 3, 4))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (2, 3, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_partially_adaptive() -> None:
    # Test case where some dimensions are adaptive and some are not
    x = torch.randn(1, 1, 8, 7, 8)
    result = adaptive_avg_pool3d(x, (4, 3, 4))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (4, 3, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_single_adaptive_dim() -> None:
    # Test where only one dimension is adaptive
    x = torch.randn(1, 2, 9, 8)
    result = adaptive_avg_pool2d(x, (3, 4))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (3, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_non_divisible_vp_multiple_channels() -> None:
    # Non-divisible variance preserving with multiple channels
    x = torch.randn(2, 4, 7, 9)
    result = adaptive_avg_pool2d(x, (3, 4), variance_preserving=True)
    assert result.shape == (2, 4, 3, 4)


def test_adaptive_avg_pool3d_single_adaptive_dim() -> None:
    # Test where only one or two dimensions are adaptive
    x = torch.randn(1, 2, 8, 9, 8)
    result = adaptive_avg_pool3d(x, (4, 3, 4))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (4, 3, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_all_adaptive() -> None:
    # Test where all dimensions are adaptive
    x = torch.randn(1, 2, 7, 9, 11)
    result = adaptive_avg_pool3d(x, (3, 4, 5))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (3, 4, 5))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_non_divisible_vp_multiple_channels() -> None:
    # Non-divisible variance preserving with multiple channels
    x = torch.randn(2, 3, 7, 9, 11)
    result = adaptive_avg_pool3d(x, (3, 4, 5), variance_preserving=True)
    assert result.shape == (2, 3, 3, 4, 5)


def test_adaptive_avg_pool2d_invalid_ndim() -> None:
    # Test error handling for invalid tensor dimensions
    x = torch.randn(4, 4)  # 2D tensor, should fail
    with pytest.raises(RuntimeError):
        adaptive_avg_pool2d(x, (2, 2))


def test_adaptive_avg_pool2d_zero_dimension() -> None:
    # Test error handling for zero-size dimensions
    x = torch.randn(1, 1, 0, 4)  # Zero height
    with pytest.raises(RuntimeError):
        adaptive_avg_pool2d(x, (2, 2))


def test_adaptive_avg_pool3d_invalid_ndim() -> None:
    # Test error handling for invalid tensor dimensions
    x = torch.randn(4, 4, 4)  # 3D tensor, should fail
    with pytest.raises(RuntimeError):
        adaptive_avg_pool3d(x, (2, 2, 2))


def test_adaptive_avg_pool3d_zero_dimension() -> None:
    # Test error handling for zero-size dimensions
    x = torch.randn(1, 1, 0, 4, 4)  # Zero depth
    with pytest.raises(RuntimeError):
        adaptive_avg_pool3d(x, (2, 2, 2))


def test_adaptive_avg_pool2d_adaptive_masking_both_dims() -> None:
    # Test the adaptive masking path for 2D pooling (lines 68-95)
    # Use input/output sizes that trigger adaptive behavior in both dimensions
    # This happens when in_size % out_size != 0 and out_size % in_size_mod != 0
    x = torch.randn(1, 2, 10, 10)
    result = adaptive_avg_pool2d(x, (3, 3))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (3, 3))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_adaptive_masking_variance_preserving() -> None:
    # Test adaptive masking with variance preserving (lines 68-95, line 94)
    # Use dimensions that trigger both adaptive=True for both height and width
    x = torch.randn(2, 3, 10, 10)
    result = adaptive_avg_pool2d(x, (3, 3), variance_preserving=True)
    assert result.shape == (2, 3, 3, 3)
    # Verify the result is not NaN or Inf
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_adaptive_avg_pool2d_adaptive_single_dim_h() -> None:
    # Test adaptive masking in height dimension only (lines 68-74)
    x = torch.randn(1, 1, 11, 8)  # 11 -> 3 is adaptive, 8 -> 4 is not
    result = adaptive_avg_pool2d(x, (3, 4))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (3, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_adaptive_single_dim_w() -> None:
    # Test adaptive masking in width dimension only (lines 75-81)
    x = torch.randn(1, 1, 8, 11)  # 8 -> 4 is not adaptive, 11 -> 3 is adaptive
    result = adaptive_avg_pool2d(x, (4, 3))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (4, 3))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_adaptive_masking_all_dims() -> None:
    # Test the adaptive masking path for 3D pooling (lines 158-196)
    # Use input/output sizes that trigger adaptive behavior in all dimensions
    x = torch.randn(1, 1, 10, 10, 10)
    result = adaptive_avg_pool3d(x, (3, 3, 3))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (3, 3, 3))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_adaptive_masking_variance_preserving() -> None:
    # Test adaptive masking with variance preserving (lines 158-196, line 194-195)
    x = torch.randn(2, 2, 10, 10, 10)
    result = adaptive_avg_pool3d(x, (3, 3, 3), variance_preserving=True)
    assert result.shape == (2, 2, 3, 3, 3)


def test_adaptive_avg_pool3d_adaptive_depth_only() -> None:
    # Test adaptive masking in depth dimension only (lines 158-164)
    x = torch.randn(1, 1, 11, 8, 8)  # 11 -> 3 is adaptive
    result = adaptive_avg_pool3d(x, (3, 4, 4))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (3, 4, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_adaptive_height_only() -> None:
    # Test adaptive masking in height dimension only (lines 165-171)
    x = torch.randn(1, 1, 8, 11, 8)  # 11 -> 3 is adaptive
    result = adaptive_avg_pool3d(x, (4, 3, 4))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (4, 3, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_adaptive_width_only() -> None:
    # Test adaptive masking in width dimension only (lines 172-178)
    x = torch.randn(1, 1, 8, 8, 11)  # 11 -> 3 is adaptive
    result = adaptive_avg_pool3d(x, (4, 4, 3))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (4, 4, 3))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_adaptive_two_dims() -> None:
    # Test adaptive masking in two dimensions
    x = torch.randn(1, 1, 11, 10, 8)
    result = adaptive_avg_pool3d(x, (3, 3, 4))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (3, 3, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_maxlength_edge_case() -> None:
    # Test case that triggers maxlength calculation edge cases (line 220)
    # When in_size % out_size == 0, maxlength should be reduced
    x = torch.randn(1, 1, 12, 12)
    result = adaptive_avg_pool2d(x, (4, 4))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_minimum_clamping() -> None:
    # Test the minimum clamping in adaptive case (lines 229-234)
    # This uses torch.minimum to clamp indices
    x = torch.randn(1, 1, 13, 13)
    result = adaptive_avg_pool2d(x, (5, 5))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (5, 5))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_length_computation() -> None:
    # Test length computation in adaptive case (lines 237-238)
    x = torch.randn(1, 1, 15, 15)
    result = adaptive_avg_pool2d(x, (7, 7))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (7, 7))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_end_index() -> None:
    # Test _end_index function usage (line 249)
    x = torch.randn(1, 1, 17, 17)
    result = adaptive_avg_pool2d(x, (6, 6))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (6, 6))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_masking_dim_neg2() -> None:
    # Test masking for dim=-2 case (lines 264-265)
    x = torch.randn(1, 2, 14, 8)
    result = adaptive_avg_pool2d(x, (5, 4))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (5, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_masking_dim_neg3() -> None:
    # Test masking for dim=-3 case in 3D pooling (lines 266-267)
    x = torch.randn(1, 2, 14, 8, 8)
    result = adaptive_avg_pool3d(x, (5, 4, 4))
    expected = torch.nn.functional.adaptive_avg_pool3d(x, (5, 4, 4))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_masked_fill() -> None:
    # Test torch.masked_fill usage (line 268)
    x = torch.randn(1, 3, 19, 19)
    result = adaptive_avg_pool2d(x, (8, 8))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (8, 8))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool2d_length_unsqueeze() -> None:
    # Test length unsqueezing (line 270)
    x = torch.randn(2, 4, 16, 16)
    result = adaptive_avg_pool2d(x, (7, 7))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (7, 7))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_complex_adaptive() -> None:
    # Complex test with multiple adaptive dimensions and variance preserving
    x = torch.randn(2, 3, 13, 15, 17)
    result = adaptive_avg_pool3d(x, (5, 6, 7), variance_preserving=True)
    assert result.shape == (2, 3, 5, 6, 7)


def test_adaptive_avg_pool2d_large_batch() -> None:
    # Test with larger batch size to ensure robustness
    x = torch.randn(8, 16, 14, 14)
    result = adaptive_avg_pool2d(x, (5, 5))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (5, 5))
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool3d_large_batch() -> None:
    # Test 3D pooling with larger batch size
    x = torch.randn(4, 8, 13, 13, 13)
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
            out[..., i, j] = block.sum(dim=(-2, -1)) / math.sqrt(count)
    return out


def test_variance_preserving_ragged_windows_keep_the_input_precision() -> None:
    """The divisor must not silently drop the result to float32.

    Ragged windows divide by a per-position int64 count, and ``count ** 0.5``
    promotes an integer tensor to float32 no matter what the input carried --
    so a float64 pool lost ~24 bits. Every other ragged test asserts only the
    shape, which is why this survived.
    """
    x = torch.randn(1, 2, 5, 5, dtype=torch.float64)
    torch.testing.assert_close(
        adaptive_avg_pool2d(x, (3, 3), variance_preserving=True),
        _reference_variance_preserving(x, (3, 3)),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_variance_preserving_returns_the_input_dtype(dtype: torch.dtype) -> None:
    """A ragged size must not change the dtype a divisible one preserves."""
    ragged = adaptive_avg_pool2d(
        torch.randn(1, 1, 5, 5, dtype=dtype),
        (3, 3),
        variance_preserving=True,
    )
    divisible = adaptive_avg_pool2d(
        torch.randn(1, 1, 8, 8, dtype=dtype),
        (4, 4),
        variance_preserving=True,
    )
    assert ragged.dtype == dtype
    assert divisible.dtype == dtype
    assert (
        adaptive_avg_pool3d(
            torch.randn(1, 1, 5, 5, 5, dtype=dtype),
            (3, 3, 3),
            variance_preserving=True,
        ).dtype
        == dtype
    )


def test_adaptive_avg_pool2d_both_dims_adaptive_vp() -> None:
    # Test to specifically cover line 94 - variance preserving with both dims adaptive
    # Using 11x11 -> 3x3:
    # 11 % 3 = 2, and 3 % 2 = 1 (not 0), so adaptive = True for both dims
    x = torch.randn(1, 1, 11, 11)
    result = adaptive_avg_pool2d(x, (3, 3), variance_preserving=True)
    assert result.shape == (1, 1, 3, 3)
    assert not torch.isnan(result).any()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
