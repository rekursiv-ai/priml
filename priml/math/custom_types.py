"""Tensor type aliases, dtype coercion, and ``convert_to_tensor``.

Lives under ``priml.math`` because every consumer is a math/data-
processing module. Carved out of ``priml.lib.custom_types`` so callers
that only need the torch-free pieces (sentinels, sequence aliases,
checkpoint/job Protocols) don't pay the ~1.2s torch + jaxtyping import
on startup.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast, overload

import functools

from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]
from torch import Tensor
from typeguard import typechecked

import numpy as np
import torch


__all__ = [
    "Numeric",
    "TensorFn",
    "TensorNest",
    "Tensorable",
    "TensorableFn",
    "TensorableNest",
    "convert_to_tensor",
    "jaxtypechecked",
]


jaxtypechecked = functools.partial(jaxtyped, typechecker=typechecked)


Numeric = bool | int | float | complex | np.number | np.bool_

# "Tensorable" follows Python's -able convention (Callable, Hashable, Iterable)
# and avoids collision with torch._prims_common.TensorLike.
Tensorable = Sequence["Tensorable"] | np.ndarray[Any, Any] | Tensor | Numeric
TensorableNest = (
    Sequence["TensorableNest"] | Mapping[str, "TensorableNest"] | Tensorable
)
TensorNest = Sequence["TensorNest"] | Mapping[str, "TensorNest"] | Tensor


type TensorFn = Callable[[Tensor], Tensor]
"""TITO -- tensor in, tensor out.

Mathematically an *endomorphism* (a map from a set into itself), or more
precisely an endofunction, since the set here is one of values rather than an
arbitrary category. Both words are avoided: they are fancier than the thing they
name, and a reader who looks one up learns nothing that ``Tensor -> Tensor`` did
not already say.
"""


type TensorableFn = Callable[[Tensorable], Tensor]
"""The usual ``priml.math`` shape: accept anything coercible, return a Tensor.

Not TITO: the domain (lists, arrays, scalars, tensors) is wider than the
codomain, so this is not a :data:`TensorFn`. Such a function composes with
itself only because ``Tensor`` is one of the things ``Tensorable`` admits. It IS
assignable where a ``TensorFn`` is wanted, since parameters are contravariant;
the reverse is rejected.
"""


@overload
def convert_to_tensor(
    __x: Tensorable,
    /,
    *,
    dtype: torch.dtype | None = ...,
    device: torch.device | str | None = ...,
    dtype_hint: torch.dtype | None = ...,
) -> Tensor: ...


@overload
def convert_to_tensor(
    __x: Tensorable,
    __y: Tensorable,
    /,
    *xs: Tensorable,
    dtype: torch.dtype | None = ...,
    device: torch.device | str | None = ...,
    dtype_hint: torch.dtype | None = ...,
) -> tuple[Tensor, ...]: ...


def convert_to_tensor(
    *xs: Tensorable,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    dtype_hint: torch.dtype | None = None,
) -> Tensor | tuple[Tensor, ...]:
    """Convert inputs to tensors with unified dtype.

    Promotes all inputs to the highest-precedence dtype found
    among them (e.g. float64 > float32 > int64). When no input
    has an explicit dtype, torch assigns defaults then unifies.

    Returns a single Tensor when called with one argument, or a
    tuple of Tensors when called with multiple arguments.

    Args:
      xs: Values convertible to tensors.
      dtype: Force all outputs to this dtype.
      device: Place all outputs on this device.
      dtype_hint: Fallback dtype when no input carries one.

    """
    if torch.compiler.is_compiling():
        for x in xs:
            assert isinstance(x, Tensor)
        # Inputs are already tensors under compile, so dtype/device are applied
        # with ``.to`` (a no-op when already matching) rather than re-tracing
        # ``torch.as_tensor``. This keeps eager and compiled paths in agreement.
        if dtype is None:
            dtype = _resolve_dtype(xs) or dtype_hint
        result = tuple(cast(Tensor, x).to(dtype=dtype, device=device) for x in xs)
        return result[0] if len(xs) == 1 else result
    if dtype is None:
        dtype = _resolve_dtype(xs)
        if dtype is None and dtype_hint is None:
            # Bare scalars: let torch assign defaults, then unify.
            xs = tuple(torch.as_tensor(x, device=device) for x in xs)
            dtype = _resolve_dtype(xs)
        elif dtype is None:
            dtype = dtype_hint
    result = tuple(torch.as_tensor(x, dtype=dtype, device=device) for x in xs)
    return result[0] if len(xs) == 1 else result


def _resolve_dtype(xs: tuple[Tensorable, ...]) -> torch.dtype | None:
    seen = set[torch.dtype]()
    for x in xs:
        dt = getattr(x, "dtype", None)
        if dt is None:
            continue
        if not isinstance(dt, torch.dtype):
            dt = _numpy_dtype_to_torch_dtype.get(dt)
            if dt is None:
                continue
        seen.add(dt)
    if not seen:
        return None
    for dtype in _dtype_coercion_precedence:
        if dtype in seen:
            return dtype
    raise ValueError(
        f"No coercible dtype among {seen}; supported dtypes (in precedence "
        f"order) are {_dtype_coercion_precedence}. Variants outside this set "
        "(e.g. complex32 or ROCm-only float8 types) must be cast explicitly.",
    )


_dtype_coercion_precedence: tuple[torch.dtype, ...] = (
    torch.complex128,
    torch.complex64,
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float8_e5m2,
    torch.float8_e4m3fn,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint64,
    torch.uint32,
    torch.uint16,
    torch.uint8,
    torch.bool,
)

_numpy_dtype_to_torch_dtype: dict[type | np.dtype[Any], torch.dtype] = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.uint16: torch.uint16,
    np.uint32: torch.uint32,
    np.uint64: torch.uint64,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
    np.dtypes.BoolDType: torch.bool,
    np.dtypes.UInt8DType: torch.uint8,
    np.dtypes.UInt16DType: torch.uint16,
    np.dtypes.UInt32DType: torch.uint32,
    np.dtypes.UInt64DType: torch.uint64,
    np.dtypes.Int8DType: torch.int8,
    np.dtypes.Int16DType: torch.int16,
    np.dtypes.Int32DType: torch.int32,
    np.dtypes.Int64DType: torch.int64,
    np.dtypes.Float16DType: torch.float16,
    np.dtypes.Float32DType: torch.float32,
    np.dtypes.Float64DType: torch.float64,
    np.dtypes.Complex64DType: torch.complex64,
    np.dtypes.Complex128DType: torch.complex128,
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("uint16"): torch.uint16,
    np.dtype("uint32"): torch.uint32,
    np.dtype("uint64"): torch.uint64,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
    np.dtypes.UShortDType: torch.uint16,
    np.dtypes.UIntDType: torch.uint32,
    np.dtypes.ULongDType: torch.uint64,
    np.dtypes.IntDType: torch.int32,
    np.dtypes.LongDType: torch.int64,
}
