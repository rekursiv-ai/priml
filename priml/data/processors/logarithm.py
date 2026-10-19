"""Binary32 logarithms, exponentials, and powers with scalar C99 evaluation."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Protocol, cast

import ctypes


if TYPE_CHECKING:
    from collections.abc import Callable

    from numba import njit
    from numpy.typing import NDArray

    import numpy as np
else:
    from wrapt import lazy_import

    njit = lazy_import("numba", "njit")
    np = lazy_import("numpy")


def log_float32(values: NDArray[np.generic]) -> NDArray[np.float32]:
    """Evaluate binary32 natural logarithms with scalar C99 rounding.

    Args:
      values: Binary32 operands of any shape, including strided and empty arrays.

    Returns:
      result: Binary32 C99 logf outputs with the original logical shape.

    Raises:
      ValueError: Operands do not have binary32 dtype.

    """
    if values.dtype != np.float32:
        raise ValueError("Expected binary32 logarithm operands.")
    return _kernel()(
        cast("_Vector", values.ravel()),
        logarithm=_libm().logf,
    ).reshape(values.shape)


def log1p_float32(values: NDArray[np.generic]) -> NDArray[np.float32]:
    """Evaluate binary32 log(1 + x) with scalar C99 rounding.

    Args:
      values: Binary32 operands of any shape, including strided and empty arrays.

    Returns:
      result: Binary32 C99 log1pf outputs with the original logical shape.

    Raises:
      ValueError: Operands do not have binary32 dtype.

    """
    if values.dtype != np.float32:
        raise ValueError("Expected binary32 logarithm operands.")
    return _kernel()(
        cast("_Vector", values.ravel()),
        logarithm=_libm().log1pf,
    ).reshape(values.shape)


def expm1_float32(values: NDArray[np.generic]) -> NDArray[np.float32]:
    """Evaluate binary32 ``exp(x) - 1`` with scalar C99 rounding.

    Args:
      values: Binary32 operands of any shape, including strided and empty arrays.

    Returns:
      result: Binary32 C99 expm1f outputs with the original logical shape.

    Raises:
      ValueError: Operands do not have binary32 dtype.

    """
    if values.dtype != np.float32:
        raise ValueError("Expected binary32 exponential operands.")
    return _kernel()(
        cast("_Vector", values.ravel()),
        logarithm=_libm().expm1f,
    ).reshape(values.shape)


def power_float32(values: NDArray[np.generic], exponent: float) -> NDArray[np.float32]:
    """Evaluate each binary32 power with scalar C99 rounding.

    Args:
      values: Binary32 bases of any shape, including strided and empty arrays.
      exponent: Shared exponent, rounded to binary32 at the C99 boundary.

    Returns:
      result: Binary32 C99 powf outputs with the original logical shape.

    Raises:
      ValueError: Bases do not have binary32 dtype.

    """
    if values.dtype != np.float32:
        raise ValueError("Expected binary32 power operands.")
    power = _libm().powf
    operands = cast("NDArray[np.float32]", values)
    return np.fromiter(
        (power(float(value), exponent) for value in operands.flat),
        dtype=np.float32,
        count=values.size,
    ).reshape(values.shape)


@cache
def _kernel() -> _LogarithmKernel:
    return cast("_LogarithmKernel", njit(cache=True)(_log_numba))


def _log_numba(
    values: _Vector,
    logarithm: Callable[[float], float],
) -> NDArray[np.float32]:
    output = np.empty(len(values), dtype=np.float32)
    for index in range(len(values)):
        # An opaque C call keeps LLVM from selecting a vector logarithm with new bits.
        output[index] = logarithm(float(values[index]))
    return output


@cache
def _libm() -> _Libm:
    library = ctypes.CDLL(None)
    library.logf.restype = ctypes.c_float
    library.logf.argtypes = [ctypes.c_float]
    library.log1pf.restype = ctypes.c_float
    library.log1pf.argtypes = [ctypes.c_float]
    library.expm1f.restype = ctypes.c_float
    library.expm1f.argtypes = [ctypes.c_float]
    library.powf.restype = ctypes.c_float
    library.powf.argtypes = [ctypes.c_float, ctypes.c_float]
    return cast("_Libm", library)


class _Libm(Protocol):
    def logf(self, x: float, /) -> float: ...

    def log1pf(self, x: float, /) -> float: ...

    def expm1f(self, x: float, /) -> float: ...

    def powf(self, x: float, y: float, /) -> float: ...


class _LogarithmKernel(Protocol):
    def __call__(
        self,
        values: _Vector,
        logarithm: Callable[[float], float],
    ) -> NDArray[np.float32]: ...


class _Vector(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> np.float32: ...
