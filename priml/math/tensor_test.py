"""Tests for core.math.tensor."""

from __future__ import annotations

import pytest
import torch

from priml.math.tensor import as_batch_tensor, pad


def test_pad_constant():
    """Test constant padding mode."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = pad(x, pad=[1, 1], mode="constant", value=0.0)
    expected = torch.tensor([0.0, 1.0, 2.0, 3.0, 0.0])
    assert torch.allclose(y, expected)


def test_pad_reflect():
    """Test reflect padding mode."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = pad(x, pad=[1, 1], mode="reflect")
    expected = torch.tensor([2.0, 1.0, 2.0, 3.0, 2.0])
    assert torch.allclose(y, expected)


def test_pad_replicate():
    """Test replicate padding mode."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = pad(x, pad=[1, 1], mode="replicate")
    expected = torch.tensor([1.0, 1.0, 2.0, 3.0, 3.0])
    assert torch.allclose(y, expected)


def test_pad_reflect_replicate_small_padding():
    """Test reflect_replicate with small padding (same as reflect)."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = pad(x, pad=[2, 2], mode="reflect_replicate")
    # With enough space, should behave like pure reflect
    expected = torch.tensor([3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0])
    assert torch.allclose(y, expected)


def test_pad_reflect_replicate_large_padding():
    """Test reflect_replicate when padding exceeds dimension."""
    # This would fail with pure reflect mode
    x = torch.tensor([1.0, 2.0])
    # Pad by 3 on each side (larger than size 2)
    y = pad(x, pad=[3, 3], mode="reflect_replicate")

    # Algorithm: For s=2, r=3:
    # c = min(3+1, max(0, 3+2-2)) // 2 = min(4, 3) // 2 = 1
    # So replicate by 1, then reflect by 2
    # After replicate by 1: [1, 1, 2, 2]
    # After reflect by 2: [1, 2, 1, 1, 2, 2, 1, 2]
    assert y.shape == (8,)


def test_pad_reflect_replicate_2d():
    """Test reflect_replicate on 2D spatial tensor."""
    # Need 3D input (batch, H, W) to pad 2 dimensions
    x = torch.ones(1, 2, 3)
    # Pad H by 2, W by 1
    y = pad(x, pad=[1, 1, 2, 2], mode="reflect_replicate")
    # Should work without error
    assert y.shape == (1, 6, 5)


def test_pad_1d_tensor_with_flatten():
    """Test padding 1D tensor (should add temporary batch dim)."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = pad(x, pad=[1, 1], mode="replicate")
    # Output should still be 1D
    assert y.ndim == 1
    assert y.shape == (5,)


def test_pad_circular():
    """Test circular padding mode."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = pad(x, pad=[1, 1], mode="circular")
    expected = torch.tensor([3.0, 1.0, 2.0, 3.0, 1.0])
    assert torch.allclose(y, expected)


def test_as_batch_tensor_add_one_dim():
    """Test as_batch_tensor adds one dimension when below min_ndim."""
    x = torch.zeros(224, 224, 3)
    result = as_batch_tensor(x, min_ndim=4, max_ndim=5)
    assert result.shape == (1, 224, 224, 3)


def test_as_batch_tensor_no_change_at_min():
    """Test as_batch_tensor unchanged when at min_ndim."""
    x = torch.zeros(2, 224, 224, 3)
    result = as_batch_tensor(x, min_ndim=4, max_ndim=5)
    assert result.shape == (2, 224, 224, 3)


def test_as_batch_tensor_no_change_at_max():
    """Test as_batch_tensor unchanged when at max_ndim."""
    x = torch.zeros(2, 8, 224, 224, 3)
    result = as_batch_tensor(x, min_ndim=4, max_ndim=5)
    assert result.shape == (2, 8, 224, 224, 3)


def test_as_batch_tensor_flatten_above_max():
    """Test as_batch_tensor flattens dimensions above max_ndim."""
    x = torch.zeros(2, 3, 8, 224, 224, 3)
    result = as_batch_tensor(x, min_ndim=4, max_ndim=5)
    assert result.shape == (6, 8, 224, 224, 3)


def test_as_batch_tensor_add_multiple_dims():
    """Test as_batch_tensor adds multiple dimensions."""
    x = torch.zeros(224, 3)
    result = as_batch_tensor(x, min_ndim=4, max_ndim=5)
    # Adds 2 dims to reach min_ndim=4
    assert result.shape == (1, 1, 224, 3)


def test_as_batch_tensor_flatten_many_dims():
    """Test as_batch_tensor flattens many dimensions."""
    x = torch.zeros(2, 3, 4, 5, 224, 224, 3)
    result = as_batch_tensor(x, min_ndim=4, max_ndim=5)
    # Flattens first 2 dims, keeps last max_ndim-1=4 dims
    assert result.shape == (24, 5, 224, 224, 3)


def test_as_batch_tensor_different_min_max():
    """Test as_batch_tensor with different min/max values."""
    x = torch.zeros(10, 20)
    result = as_batch_tensor(x, min_ndim=3, max_ndim=4)
    # Adds 1 dim to reach min_ndim=3
    assert result.shape == (1, 10, 20)


def test_as_batch_tensor_preserves_values():
    """Test as_batch_tensor preserves tensor values."""
    x = torch.arange(24).reshape(2, 3, 4)
    result = as_batch_tensor(x, min_ndim=4, max_ndim=5)
    assert result.shape == (1, 2, 3, 4)
    assert torch.all(result[0] == x)


def test_as_batch_tensor_flatten_preserves_values():
    """Test as_batch_tensor flattening preserves values."""
    x = torch.arange(24).reshape(2, 3, 4)
    result = as_batch_tensor(x, min_ndim=2, max_ndim=2)
    assert result.shape == (6, 4)
    assert torch.all(result.reshape(-1) == x.reshape(-1))


def test_as_batch_tensor_with_device():
    """Test as_batch_tensor respects device parameter."""
    x = torch.zeros(224, 224, 3)
    result = as_batch_tensor(x, min_ndim=4, max_ndim=5, device="cpu")
    assert result.device.type == "cpu"


def test_as_batch_tensor_with_dtype():
    """Test as_batch_tensor respects dtype parameter."""
    x = torch.zeros(224, 224, 3, dtype=torch.uint8)
    result = as_batch_tensor(x, min_ndim=4, max_ndim=5, dtype=torch.float32)
    assert result.dtype == torch.float32


def test_as_batch_tensor_edge_case_scalar():
    """Test as_batch_tensor with scalar-like tensor."""
    x = torch.tensor(5.0)
    result = as_batch_tensor(x, min_ndim=2, max_ndim=3)
    assert result.shape == (1, 1)


def test_as_batch_tensor_min_equals_max():
    """Test as_batch_tensor when min_ndim equals max_ndim."""
    x = torch.zeros(2, 3, 4)
    result = as_batch_tensor(x, min_ndim=3, max_ndim=3)
    assert result.shape == (2, 3, 4)


def test_as_batch_tensor_large_flatten():
    """Test as_batch_tensor with large batch flattening."""
    x = torch.zeros(2, 3, 4, 5, 6)
    result = as_batch_tensor(x, min_ndim=2, max_ndim=3)
    # Keeps last max_ndim-1=2 dims, flattens first 3
    assert result.shape == (24, 5, 6)


def test_as_batch_tensor_list_input():
    """Test as_batch_tensor accepts list input."""
    x = [[[1, 2, 3]]]
    result = as_batch_tensor(x, min_ndim=4, max_ndim=5)
    assert result.shape == (1, 1, 1, 3)


def test_as_batch_tensor_negative_min_raises():
    """Test as_batch_tensor raises on negative min_ndim."""
    x = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="Nonpositive min_ndim"):
        as_batch_tensor(x, min_ndim=-1, max_ndim=3)


def test_as_batch_tensor_negative_max_raises():
    """Test as_batch_tensor raises on negative max_ndim."""
    x = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="Negative max_ndim"):
        as_batch_tensor(x, min_ndim=2, max_ndim=-1)


def test_as_batch_tensor_max_less_than_min_below_both():
    """Test as_batch_tensor when max_ndim < min_ndim and x.ndim < max_ndim."""
    x = torch.zeros(10, 20)
    result = as_batch_tensor(x, min_ndim=5, max_ndim=3)
    # x.ndim=2 < max_ndim=3 < min_ndim=5
    # c=1, d=3, a=max(0,min(1,3)-1)=0, b=max(0,0,-2)=0
    # reshape(-1, *[], *x.shape[0:]) = (1, 10, 20)
    assert result.shape == (1, 10, 20)


def test_as_batch_tensor_max_less_than_min_between():
    """Test as_batch_tensor when max_ndim < min_ndim and max_ndim < x.ndim < min_ndim."""
    x = torch.zeros(2, 3, 4, 5)
    result = as_batch_tensor(x, min_ndim=6, max_ndim=3)
    # max_ndim=3 < x.ndim=4 < min_ndim=6
    # c=-1, d=2, a=max(0,min(-1,2)-1)=0, b=max(0,2,-1)=2
    # reshape(-1, *[], *x.shape[2:]) = (6, 4, 5)
    assert result.shape == (6, 4, 5)


def test_as_batch_tensor_max_less_than_min_above_both():
    """Test as_batch_tensor when max_ndim < min_ndim and x.ndim > min_ndim."""
    x = torch.zeros(2, 3, 4, 5, 6, 7, 8)
    result = as_batch_tensor(x, min_ndim=5, max_ndim=3)
    # max_ndim=3 < min_ndim=5 < x.ndim=7
    # Should flatten to keep only max_ndim-1=2 trailing dims
    assert result.shape == (720, 7, 8)


def test_as_batch_tensor_max_less_than_min_at_max():
    """Test as_batch_tensor when max_ndim < min_ndim and x.ndim == max_ndim."""
    x = torch.zeros(10, 20, 30)
    result = as_batch_tensor(x, min_ndim=5, max_ndim=3)
    # x.ndim=3 == max_ndim=3 < min_ndim=5
    # c=0, d=2, a=max(0,min(0,2)-1)=0, b=max(0,1,-1)=1
    # reshape(-1, *[], *x.shape[1:]) = (10, 20, 30) - unchanged
    assert result.shape == (10, 20, 30)


def test_as_batch_tensor_max_less_than_min_at_min():
    """Test as_batch_tensor when max_ndim < min_ndim and x.ndim == min_ndim."""
    x = torch.zeros(2, 3, 4, 5, 6)
    result = as_batch_tensor(x, min_ndim=5, max_ndim=3)
    # max_ndim=3 < x.ndim=5 == min_ndim=5
    # Should flatten to keep max_ndim-1=2 trailing dims
    assert result.shape == (24, 5, 6)


def test_as_batch_tensor_max_less_than_min_preserves_values():
    """Test as_batch_tensor preserves values when max_ndim < min_ndim."""
    x = torch.arange(24).reshape(2, 3, 4)
    result = as_batch_tensor(x, min_ndim=6, max_ndim=4)
    # x.ndim=3 < max_ndim=4 < min_ndim=6
    # c=1, d=3, a=max(0,min(1,3)-1)=0, b=max(0,0,-2)=0
    # reshape(-1, *[], *x.shape[0:]) = (1, 2, 3, 4)
    assert result.shape == (1, 2, 3, 4)
    assert torch.all(result[0] == x)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
