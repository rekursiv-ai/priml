"""Pin binary32 logarithm rounding independently of NumPy CPU dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from priml.math import logarithm


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture
def eager_kernel(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Execute the same C99 calls without compiling ordinary unit tests."""
    logarithm._kernel.cache_clear()
    monkeypatch.setattr(logarithm, "njit", _eager_compile)
    yield
    logarithm._kernel.cache_clear()


@pytest.mark.usefixtures("eager_kernel")
@pytest.mark.parametrize("transpose", [False, True])
def test_natural_logarithm_preserves_source_scalar_rounding(transpose: bool) -> None:
    values = np.array(
        [[0x3F7F_F026, 0x3F7F_FFF4, 0x3F7F_FFFE], [0x3F80_0000, 0x3F00_0000, 0]],
        dtype=np.uint32,
    ).view(np.float32)
    expected = np.array(
        [[3_112_019_930, 3_040_870_404, 3_019_898_880], [0, 0xBF31_7218, 0xFF80_0000]],
        dtype=np.uint32,
    ).view(np.float32)
    if transpose:
        values, expected = values.T, expected.T
    original = values.copy()
    actual = logarithm.log_float32(values)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(values.view(np.uint32), original.view(np.uint32))
    assert actual.dtype == np.float32
    assert actual.shape == values.shape


@pytest.mark.usefixtures("eager_kernel")
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3)])
def test_natural_logarithm_preserves_scalar_and_empty_shapes(
    shape: tuple[int, ...],
) -> None:
    values = np.ones(shape, dtype=np.float32)
    actual = logarithm.log_float32(values)
    np.testing.assert_array_equal(actual, np.zeros(shape, dtype=np.float32))
    assert actual.dtype == np.float32
    assert actual.shape == shape


@pytest.mark.usefixtures("eager_kernel")
def test_natural_logarithm_propagates_domain_and_nonfinite_results() -> None:
    values = np.array([-1, -0.0, np.inf, np.nan], dtype=np.float32)
    expected = np.array([np.nan, -np.inf, np.inf, np.nan], dtype=np.float32)
    np.testing.assert_array_equal(logarithm.log_float32(values), expected)


@pytest.mark.usefixtures("eager_kernel")
@pytest.mark.parametrize("dtype", [np.float64, np.int32])
def test_natural_logarithm_rejects_non_binary32_metadata(
    dtype: type[np.generic],
) -> None:
    with pytest.raises(ValueError, match="binary32"):
        logarithm.log_float32(np.ones(3, dtype=dtype))


def _eager_compile(
    *,
    cache: bool,
) -> Callable[[logarithm._LogarithmKernel], logarithm._LogarithmKernel]:
    del cache
    return lambda kernel: kernel


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
