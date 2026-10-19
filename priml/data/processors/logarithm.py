"""Binary32 logarithms with explicit scalar C99 evaluation."""

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
    """Evaluate binary32 log(1 + x) with fdlibm's ``log1pf`` rounding.

    The captured references ran fdlibm's algorithm (as must and older glibc
    do); newer glibc rounds correctly and differs by one ulp on some inputs, so
    the algorithm is evaluated here rather than taken from the host's libm.

    Args:
      values: Binary32 operands of any shape, including strided and empty arrays.

    Returns:
      result: Binary32 fdlibm log1pf outputs with the original logical shape.

    Raises:
      ValueError: Operands do not have binary32 dtype.

    """
    if values.dtype != np.float32:
        raise ValueError("Expected binary32 logarithm operands.")
    return _log1p_kernel()(cast("_Vector", values.ravel())).reshape(values.shape)


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
def _log1p_kernel() -> Callable[[_Vector], NDArray[np.float32]]:
    scalar = njit(cache=True)(_log1pf)

    def kernel(values: _Vector) -> NDArray[np.float32]:
        output = np.empty(len(values), dtype=np.float32)
        for index in range(len(values)):
            output[index] = scalar(values[index])
        return output

    return cast("Callable[[_Vector], NDArray[np.float32]]", njit(cache=True)(kernel))


# The fdlibm s_log1pf.c (Sun Microsystems, 1993; freely redistributable), with every
# intermediate rounded to binary32 as the C does. The constants are its float32
# words, written as the exact doubles they widen to.
def _log1pf(x: np.float32) -> np.float32:
    f32 = np.float32
    hx = int(np.array([x], dtype=np.float32).view(np.int32).sum())
    ax = hx & 0x7FFFFFFF
    k, hu, f, c = 1, 0, f32(0.0), f32(0.0)
    if hx < 0x3ED413D7:
        if ax >= 0x3F800000:
            return f32(-np.inf) if x == f32(-1.0) else f32(np.nan)
        if ax < 0x31000000:
            return x if ax < 0x24800000 else f32(x - f32(f32(x * x) * f32(0.5)))
        if hx > 0 or hx <= -1_097_468_385:
            k, f, hu = 0, x, 1
    elif hx >= 0x7F800000:
        return f32(x + x)
    if k != 0:
        u = f32(f32(1.0) + x) if hx < 0x5A000000 else x
        hu = int(np.array([u], dtype=np.float32).view(np.int32).sum())
        k = (hu >> 23) - 127
        if hx < 0x5A000000:
            c = f32(f32(1.0) - f32(u - x)) if k > 0 else f32(x - f32(u - f32(1.0)))
            c = f32(c / u)
        hu &= 0x007FFFFF
        if hu < 0x3504F7:
            u = f32(np.array([hu | 0x3F800000], dtype=np.int32).view(f32).sum())
        else:
            k += 1
            u = f32(np.array([hu | 0x3F000000], dtype=np.int32).view(f32).sum())
            hu = (0x00800000 - hu) >> 2
        f = f32(u - f32(1.0))
    hfsq = f32(f32(f32(0.5) * f) * f)
    fk = f32(k)
    ln2_hi, ln2_lo = f32(0.6931381225585938), f32(9.05800061445916e-06)
    # In binary32 a zero mantissa leaves f == 0 and k != 0 (k == 0 set hu = 1):
    # 1 + x is an exact power of two, and fdlibm's other hu == 0 arms are dead.
    if hu == 0:
        return f32(f32(fk * ln2_hi) + f32(c + f32(fk * ln2_lo)))
    s = f32(f / f32(f32(2.0) + f))
    z = f32(s * s)
    r = f32(0.14798198640346527)
    r = f32(f32(0.15313838422298431) + f32(z * r))
    r = f32(f32(0.18183572590351105) + f32(z * r))
    r = f32(f32(0.2222219854593277) + f32(z * r))
    r = f32(f32(0.2857142984867096) + f32(z * r))
    r = f32(f32(0.4000000059604645) + f32(z * r))
    r = f32(f32(0.6666666865348816) + f32(z * r))
    r = f32(z * r)
    if k == 0:
        return f32(f - f32(hfsq - f32(s * f32(hfsq + r))))
    tail = f32(f32(s * f32(hfsq + r)) + f32(f32(fk * ln2_lo) + c))
    return f32(f32(fk * ln2_hi) - f32(f32(hfsq - tail) - f))


@cache
def _libm() -> _Libm:
    library = ctypes.CDLL(None)
    library.logf.restype = ctypes.c_float
    library.logf.argtypes = [ctypes.c_float]
    return cast("_Libm", library)


class _Libm(Protocol):
    def logf(self, x: float, /) -> float: ...


class _LogarithmKernel(Protocol):
    def __call__(
        self,
        values: _Vector,
        logarithm: Callable[[float], float],
    ) -> NDArray[np.float32]: ...


class _Vector(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> np.float32: ...
