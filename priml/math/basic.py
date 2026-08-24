from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, cast, overload

import math

from torch import Tensor

import numpy as np
import torch

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
    """Ceiling of x / y, for divisors of either sign.

    ``(x + y - 1) // y`` is the familiar form but it holds only for positive
    ``y``: it returns 4 for ``ceil_div(-5, -2)`` where the ceiling is 3, and -1
    for ``ceil_div(5, -2)`` where it is -2. Negating floor division is exact
    for every sign.
    """
    return -(-x // y)


class SupportsLT(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...


def argsort(
    x: Sequence[SupportsLT],
    descending: bool = False,
) -> list[int]:
    """Indices that would sort x, stably.

    Args:
      x: Sequence of comparable values.
      descending: If True, sort in descending order.

    Returns:
      indices: Sorted index list.

    """
    return sorted(range(len(x)), key=x.__getitem__, reverse=descending)


def broadcast_sequences[T](*args: T | Sequence[T]) -> tuple[list[T], ...]:
    """Broadcast scalar-or-sequence arguments to matching lengths.

    Each argument is either a scalar (wrapped in a list) or a sequence.
    All sequences must have the same length, or length 1 (broadcast).

    ``Sequence``, not ``Iterable``: a ``Tensor`` and a ``str`` are both
    iterable and both are meant to arrive here as scalars, and an exhausted
    generator cannot be re-read by the caller that passed it. A generator
    was wrapped as ONE element and failed downstream at ``int(<generator>)``
    -- pass a list.

    Args:
      *args: Scalars or sequences to broadcast.

    Returns:
      result: Tuple of lists, all with the same length.

    Raises:
      ValueError: If two sequences have different lengths and neither is 1.

    """
    # Cast rather than narrowed by ``isinstance``: ``T`` may itself be a
    # Sequence, so the check cannot tell the two arms of the union apart and
    # both checkers widen the element type the signature already stated.
    lists: list[list[T]] = [
        list(cast("Sequence[T]", a)) if isinstance(a, Sequence) else [a] for a in args
    ]
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
    # An integer ``multiple`` puts every result on an integer grid, so the
    # result is integral whatever came in. A float ``multiple`` does not: the
    # grid runs through 2.5, and casting an INTEGER input back to its own dtype
    # returned 2 -- not a multiple of 2.5 at all. So a float grid keeps a float
    # input's width and leaves an integer one to promote, as torch and numpy
    # would unaided.
    if isinstance(x, Tensor):
        scaled = (x / multiple).ceil() if up else (x / multiple).floor()
        snapped = multiple * scaled
        if isinstance(multiple, int):
            return snapped.to(torch.int64)
        return snapped.to(x.dtype) if x.dtype.is_floating_point else snapped
    if isinstance(x, np.ndarray):
        ratio = x / multiple
        scaled = np.ceil(ratio) if up else np.floor(ratio)
        snapped = multiple * scaled
        if isinstance(multiple, int):
            return snapped.astype(np.int64)
        return (
            snapped.astype(x.dtype) if np.issubdtype(x.dtype, np.floating) else snapped
        )
    if not isinstance(x, (int, float)):
        # A Sequence is in ``Tensorable`` and reaches here, so this is caller
        # input rather than an internal invariant: it raises.
        raise TypeError(f"Unsupported scalar type {type(x).__name__}.")
    # Two ints stay in unbounded integer arithmetic: float(x) drops the low
    # bits above 2**53, returning a value below one already on the grid.
    if isinstance(x, int) and isinstance(multiple, int):
        return multiple * (ceil_div(x, multiple) if up else x // multiple)
    value = x / multiple
    scaled_scalar = math.ceil(value) if up else math.floor(value)
    return type(multiple)(multiple * scaled_scalar)
