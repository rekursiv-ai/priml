from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, overload

import math

from torch import Tensor

import numpy as np

from priml.math.custom_types import Tensorable


def factors(n: int) -> tuple[int, ...]:
    """Integer factors of n, sorted ascending.

    Args:
      n: A positive integer.

    Returns:
      factors: Tuple of all positive integer divisors of n.

    Raises:
      ValueError: If n is not a positive integer.

    """
    if n < 1:
        raise ValueError(f"factors requires a positive integer, got {n}.")
    divisors: list[int] = []
    for i in range(1, math.isqrt(n) + 1):
        if n % i != 0:
            continue
        divisors.append(i)
        d = n // i
        if d != i:
            divisors.append(d)
    return tuple(sorted(divisors))


@overload
def ceil_multiple(x: float, multiple: int) -> int: ...
@overload
def ceil_multiple(x: float, multiple: float) -> float: ...
@overload
def ceil_multiple(x: float, multiple: None) -> float: ...
@overload
def ceil_multiple(x: Tensorable, multiple: float | None) -> Tensorable: ...
def ceil_multiple(x: Tensorable, multiple: float | None) -> Tensorable:
    """Return the smallest multiple of ``multiple`` that is >= ``x``.

    ``multiple`` is the grid spacing (step), not a number base. ``None``
    leaves ``x`` unchanged. The result type follows ``multiple``: an integer
    ``multiple`` yields an integer-typed result. Exact multiples are returned
    unchanged (idempotent), and ``multiple * ceil(x / multiple)`` avoids the
    integer-floor representation error of the ``(x + m - 1) // m`` trick.
    """
    return _to_multiple(x, multiple, up=True)


@overload
def floor_multiple(x: float, multiple: int) -> int: ...
@overload
def floor_multiple(x: float, multiple: float) -> float: ...
@overload
def floor_multiple(x: float, multiple: None) -> float: ...
@overload
def floor_multiple(x: Tensorable, multiple: float | None) -> Tensorable: ...
def floor_multiple(x: Tensorable, multiple: float | None) -> Tensorable:
    """Return the largest multiple of ``multiple`` that is <= ``x``.

    ``multiple`` is the grid spacing (step), not a number base. ``None``
    leaves ``x`` unchanged. The result type follows ``multiple``: an integer
    ``multiple`` yields an integer-typed result. Exact multiples are returned
    unchanged (idempotent).
    """
    return _to_multiple(x, multiple, up=False)


def ceil_div(x: int, y: int) -> int:
    """Ceiling of x / y."""
    return (x + y - 1) // y


class SupportsLT(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...


def argsort(
    x: Sequence[SupportsLT],
    descending: bool = False,
) -> list[int]:
    """Lexicographic argsort: indices that would sort x.

    Args:
      x: Sequence of comparable values.
      descending: If True, sort in descending order.

    Returns:
      indices: Sorted index list.

    """
    return sorted(range(len(x)), key=x.__getitem__, reverse=descending)


def broadcast_sequences(
    *args: Any | Sequence[Any],
) -> tuple[list[Any], ...]:
    """Broadcast scalar-or-sequence arguments to matching lengths.

    Each argument is either a scalar (wrapped in a list) or a sequence.
    All sequences must have the same length, or length 1 (broadcast).

    Args:
      *args: Scalars or sequences to broadcast.

    Returns:
      result: Tuple of lists, all with the same length.

    """
    lists = [[a] if not isinstance(a, Sequence) else list(a) for a in args]  # pyright: ignore[reportUnknownArgumentType]
    n = max(len(a) for a in lists)
    for a in lists:
        if len(a) not in (1, n):
            lengths = [len(x) for x in lists]
            raise ValueError(f"Incompatible lengths: {lengths}.")
    return tuple(a * n if len(a) == 1 else a for a in lists)


def _to_multiple(
    x: Tensorable,
    multiple: float | None,
    up: bool,
) -> Tensorable:
    """``multiple * (ceil|floor)(x / multiple)``, cast back to ``multiple``'s type.

    ``up`` selects ceiling (True) or floor (False). ``None`` leaves ``x``
    unchanged. Float division + ceil/floor (rather than the integer ``//``
    trick) is correct for fractional and negative ``x`` and idempotent on
    exact multiples; the final cast makes an integer ``multiple`` yield an
    integer result. Each branch narrows ``x`` before arithmetic so the result
    type is concrete (no operations on the bare ``Tensorable`` union).
    """
    if multiple is None:
        return x
    if isinstance(x, Tensor):
        scaled = (x / multiple).ceil() if up else (x / multiple).floor()
        # tensor.to(int|float) maps to the int64/float32 dtype at runtime; the
        # torch stub's .to() overloads don't list the Python-type form.
        return (multiple * scaled).to(type(multiple))  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownVariableType] -- stub gap: .to(int) valid at runtime
    if isinstance(x, np.ndarray):
        ratio = x / multiple
        scaled = np.ceil(ratio) if up else np.floor(ratio)
        return (multiple * scaled).astype(type(multiple))
    # Scalar path: Tensorable nominally includes Sequence and complex, neither
    # of which is a real runtime argument to these grid-rounding helpers.
    value = float(x) / multiple  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- real arg is int|float, not Sequence/complex
    scaled_scalar = math.ceil(value) if up else math.floor(value)
    return type(multiple)(multiple * scaled_scalar)
