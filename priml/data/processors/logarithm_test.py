"""Pin binary32 logarithm rounding independently of NumPy CPU dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

import numpy as np
import pytest

from priml.data.processors import logarithm


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture
def eager_kernel(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Execute the same scalar code without compiling ordinary unit tests."""
    logarithm._kernel.cache_clear()
    logarithm._log1p_kernel.cache_clear()
    monkeypatch.setattr(logarithm, "njit", _eager_compile)
    yield
    logarithm._kernel.cache_clear()
    logarithm._log1p_kernel.cache_clear()


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
    with pytest.raises(ValueError, match=r"^Expected binary32 logarithm operands\.$"):
        logarithm.log_float32(np.ones(3, dtype=dtype))


@pytest.mark.usefixtures("eager_kernel")
@pytest.mark.parametrize("transpose", [False, True])
def test_log1p_preserves_captured_scalar_rounding(transpose: bool) -> None:
    # Scalar source captures from the CASMI and ROGII reference processes.
    values = np.array(
        [[0, 0.46771079301834106, 1e-30], [0.5, 0.9823818206787109, -0.0]],
        dtype=np.float32,
    )
    expected = np.array(
        [
            [0, 0.38370388746261597, 1e-30],
            [0.40546509623527527, 0.6842990517616272, -0.0],
        ],
        dtype=np.float32,
    )
    if transpose:
        values, expected = values.T, expected.T
    original = values.copy()
    actual = logarithm.log1p_float32(values)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(values.view(np.uint32), original.view(np.uint32))
    assert actual.dtype == np.float32
    assert actual.shape == values.shape


@pytest.mark.usefixtures("eager_kernel")
def test_log1p_does_not_depend_on_the_hosts_libm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Glibc 2.43 rounds log1pf(0.46771079) correctly, one ulp above the capture.
    monkeypatch.setattr(logarithm, "_libm", lambda: _CorrectlyRounded)
    values = np.array([0.46771079301834106], dtype=np.float32)
    expected = np.array([1_053_062_358], dtype=np.uint32)
    np.testing.assert_array_equal(
        logarithm.log1p_float32(values).view(np.uint32),
        expected,
    )


@pytest.mark.usefixtures("eager_kernel")
@pytest.mark.parametrize(
    "x",
    [
        *(x / 4 for x in range(-3, 256, 4)),
        -0.999,
        -0.75,
        -0.9998779296875,
        2.0**30 - 1,
        2.0**30,
        2.0**-20,
        1e-9,
        -1e-9,
        0.41,
        -0.29,
        1e20,
        3e38,
    ],
)
def test_log1p_is_within_one_ulp_of_the_exact_value(x: float) -> None:
    exact = np.array([math.log1p(float(np.float32(x)))], dtype=np.float32)
    got = logarithm.log1p_float32(np.array([x], dtype=np.float32))
    assert abs(int(got.view(np.int32).sum()) - int(exact.view(np.int32).sum())) <= 1


@pytest.mark.usefixtures("eager_kernel")
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3)])
def test_log1p_preserves_scalar_and_empty_shapes(shape: tuple[int, ...]) -> None:
    actual = logarithm.log1p_float32(np.zeros(shape, dtype=np.float32))
    np.testing.assert_array_equal(actual, np.zeros(shape, dtype=np.float32))
    assert actual.dtype == np.float32
    assert actual.shape == shape


@pytest.mark.usefixtures("eager_kernel")
def test_log1p_propagates_domain_and_nonfinite_results() -> None:
    values = np.array([-2, -1, np.inf, np.nan], dtype=np.float32)
    expected = np.array([np.nan, -np.inf, np.inf, np.nan], dtype=np.float32)
    np.testing.assert_array_equal(logarithm.log1p_float32(values), expected)


@pytest.mark.usefixtures("eager_kernel")
@pytest.mark.parametrize("dtype", [np.float64, np.int32])
def test_log1p_rejects_non_binary32_metadata(dtype: type[np.generic]) -> None:
    with pytest.raises(ValueError, match=r"^Expected binary32 logarithm operands\.$"):
        logarithm.log1p_float32(np.ones(3, dtype=dtype))


@pytest.mark.compute_large_fixture
def test_compiled_logarithms_preserve_captured_scalar_rounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logarithm._kernel.cache_clear()
    logarithm._log1p_kernel.cache_clear()
    try:
        for transpose in (False, True):
            test_natural_logarithm_preserves_source_scalar_rounding(transpose)
            test_log1p_preserves_captured_scalar_rounding(transpose)
        with monkeypatch.context() as mutation:
            mutation.setattr(logarithm, "_log1pf", _identity_log1pf)
            logarithm._log1p_kernel.cache_clear()
            with pytest.raises(AssertionError):
                test_log1p_preserves_captured_scalar_rounding(False)
    finally:
        logarithm._kernel.cache_clear()
        logarithm._log1p_kernel.cache_clear()


class _CorrectlyRounded:
    """A host libm whose log1pf rounds correctly, as glibc 2.43's does."""

    @classmethod
    def logf(cls, x: float, /) -> float:
        return float(np.float32(math.log(x)))

    @classmethod
    def log1pf(cls, x: float, /) -> float:
        return float(np.float32(math.log1p(x)))


def _identity_log1pf(x: np.float32) -> np.float32:
    return x


def _eager_compile(
    *,
    cache: bool,
) -> Callable[[logarithm._LogarithmKernel], logarithm._LogarithmKernel]:
    del cache
    return lambda kernel: kernel


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
