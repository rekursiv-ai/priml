# pyright: reportPrivateUsage=false
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from priml.math import custom_types
from priml.math.custom_types import convert_to_tensor


def test_single_python_scalar():
    result = convert_to_tensor(42)
    assert isinstance(result, torch.Tensor)
    assert result.item() == 42


def test_multiple_python_scalars():
    result = convert_to_tensor(1, 2.5, True)
    assert len(result) == 3
    torch.testing.assert_close(result[0], torch.tensor(1.0))
    torch.testing.assert_close(result[1], torch.tensor(2.5))
    torch.testing.assert_close(result[2], torch.tensor(1.0))


def test_numpy_array():
    x = np.array([1, 2, 3], dtype=np.int32)
    result = convert_to_tensor(x)
    assert result.dtype == torch.int32
    torch.testing.assert_close(result, torch.tensor([1, 2, 3], dtype=torch.int32))


def test_torch_tensor():
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    result = convert_to_tensor(x)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, x)


@pytest.mark.parametrize(
    ("dtype1", "dtype2", "expected_dtype"),
    [
        (torch.complex128, torch.float32, torch.complex128),
        (torch.complex64, torch.float64, torch.complex64),
        (torch.float64, torch.float32, torch.float64),
        (torch.float32, torch.int64, torch.float32),
        (torch.float16, torch.int32, torch.float16),
        (torch.bfloat16, torch.int16, torch.bfloat16),
        (torch.int64, torch.int32, torch.int64),
        (torch.int32, torch.uint64, torch.int32),
        (torch.int16, torch.uint32, torch.int16),
        (torch.uint64, torch.uint32, torch.uint64),
        (torch.uint32, torch.uint16, torch.uint32),
        (torch.uint16, torch.uint8, torch.uint16),
        (torch.uint8, torch.bool, torch.uint8),
    ],
)
def test_type_coercion_precedence(
    dtype1: torch.dtype,
    dtype2: torch.dtype,
    expected_dtype: torch.dtype,
) -> None:
    x = torch.tensor([1], dtype=dtype1)
    y = torch.tensor([2], dtype=dtype2)
    result = convert_to_tensor(x, y)
    assert result[0].dtype == expected_dtype
    assert result[1].dtype == expected_dtype


def test_explicit_dtype_override():
    x = torch.tensor([1], dtype=torch.int32)
    y = torch.tensor([2.5], dtype=torch.float64)
    result = convert_to_tensor(x, y, dtype=torch.float32)
    assert result[0].dtype == torch.float32
    assert result[1].dtype == torch.float32


@pytest.mark.gpu_torch_cuda
def test_device_parameter():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x = torch.tensor([1, 2, 3])
    result = convert_to_tensor(x, device="cuda")
    assert result[0].device.type == "cuda"


def test_dtype_hint_with_python_scalars():
    result = convert_to_tensor(1, 2, 3, dtype_hint=torch.float64)
    assert result[0].dtype == torch.float64
    assert result[1].dtype == torch.float64
    assert result[2].dtype == torch.float64


def test_dtype_hint_ignored_when_dtype_present():
    x = torch.tensor([1], dtype=torch.float32)
    result = convert_to_tensor(x, dtype_hint=torch.float64)
    assert result[0].dtype == torch.float32


def test_mixed_numpy_and_torch():
    x = np.array([1, 2], dtype=np.float32)
    y = torch.tensor([3, 4], dtype=torch.float64)
    result = convert_to_tensor(x, y)
    assert result[0].dtype == torch.float64
    assert result[1].dtype == torch.float64


def test_complex_numbers():
    x = 1 + 2j
    result = convert_to_tensor(x)
    assert result.dtype == torch.complex64
    assert result.item() == (1 + 2j)


def test_bfloat16_dtype():
    x = torch.tensor([1.0], dtype=torch.bfloat16)
    y = torch.tensor([2], dtype=torch.int32)
    result = convert_to_tensor(x, y)
    assert result[0].dtype == torch.bfloat16
    assert result[1].dtype == torch.bfloat16


def test_float8_e5m2_dtype():
    if not hasattr(torch, "float8_e5m2"):
        pytest.skip("float8_e5m2 not available in this PyTorch version")
    x = torch.tensor([1.0], dtype=torch.float8_e5m2)
    y = torch.tensor([2], dtype=torch.int32)
    result = convert_to_tensor(x, y)
    assert result[0].dtype == torch.float8_e5m2
    assert result[1].dtype == torch.float8_e5m2


def test_float8_e4m3fn_dtype():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("float8_e4m3fn not available in this PyTorch version")
    x = torch.tensor([1.0], dtype=torch.float8_e4m3fn)
    y = torch.tensor([2], dtype=torch.int32)
    result = convert_to_tensor(x, y)
    assert result[0].dtype == torch.float8_e4m3fn
    assert result[1].dtype == torch.float8_e4m3fn


def test_nested_sequence():
    x = [[1, 2], [3, 4]]
    result = convert_to_tensor(x)
    assert result.shape == (2, 2)
    torch.testing.assert_close(result, torch.tensor([[1, 2], [3, 4]]))


@pytest.mark.parametrize(
    ("np_dtype", "expected_torch_dtype"),
    [
        (np.bool_, torch.bool),
        (np.uint8, torch.uint8),
        (np.uint16, torch.uint16),
        (np.uint32, torch.uint32),
        (np.uint64, torch.uint64),
        (np.int8, torch.int8),
        (np.int16, torch.int16),
        (np.int32, torch.int32),
        (np.int64, torch.int64),
        (np.float16, torch.float16),
        (np.float32, torch.float32),
        (np.float64, torch.float64),
        (np.complex64, torch.complex64),
        (np.complex128, torch.complex128),
    ],
)
def test_numpy_dtype_mapping(
    np_dtype: type[np.generic],
    expected_torch_dtype: torch.dtype,
) -> None:
    x = np.array([1, 2, 3], dtype=np_dtype)
    result = convert_to_tensor(x)
    assert result[0].dtype == expected_torch_dtype


def test_bool_dtype():
    x = torch.tensor([True, False], dtype=torch.bool)
    y = torch.tensor([1, 0], dtype=torch.int32)
    result = convert_to_tensor(x, y)
    assert result[0].dtype == torch.int32
    assert result[1].dtype == torch.int32


def test_as_tensors_while_compiling():
    """Test convert_to_tensor behavior during compilation (lines 147-149)."""
    # Simulate compilation context
    with patch.object(torch.compiler, "is_compiling", return_value=True):
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])
        result = convert_to_tensor(x, y)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)


def test_compile_honors_dtype():
    """Under the compile path, an explicit dtype must be applied, not dropped.

    Regression: the compile branch returned inputs unchanged, so eager and
    traced execution disagreed on output dtype. ``is_compiling`` is patched
    (rather than invoking ``torch.compile``) to exercise the branch directly.
    """
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    eager = convert_to_tensor(x, dtype=torch.float64)
    with patch.object(torch.compiler, "is_compiling", return_value=True):
        compiled = convert_to_tensor(x, dtype=torch.float64)
    assert eager.dtype == torch.float64
    assert compiled.dtype == torch.float64
    torch.testing.assert_close(compiled, eager)


def test_compile_honors_device():
    """Under the compile path, an explicit device must be applied."""
    with patch.object(torch.compiler, "is_compiling", return_value=True):
        result = convert_to_tensor(torch.tensor([1.0, 2.0]), device="cpu")
    assert result.device.type == "cpu"


def test_compile_resolves_dtype_when_unspecified():
    """No explicit dtype under compile still unifies inputs by precedence."""
    x = torch.tensor([1], dtype=torch.int32)
    y = torch.tensor([2.0], dtype=torch.float64)
    with patch.object(torch.compiler, "is_compiling", return_value=True):
        a, b = convert_to_tensor(x, y)
    assert a.dtype == torch.float64
    assert b.dtype == torch.float64


def test_as_tensors_unrecognized_dtype():
    """``_resolve_dtype`` raises when no precedence entry matches."""
    x = torch.tensor([1.0], dtype=torch.float32)
    y = torch.tensor([2.0], dtype=torch.float64)

    original_precedence = custom_types._dtype_coercion_precedence
    try:
        custom_types._dtype_coercion_precedence = ()
        with pytest.raises(ValueError, match="No coercible dtype"):
            convert_to_tensor(x, y)
    finally:
        custom_types._dtype_coercion_precedence = original_precedence


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
