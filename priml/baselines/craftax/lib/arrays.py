"""numpy arrays typed where numpy's stubs leave them ``Any``.

numpy types an array's extents as ``tuple[Any, ...]``, and an archive member, a
structured field or an untyped return as an array of unknown elements, so every
extent and element read from them is ``Any``. :data:`Shaped` types the extents
as ints, and :func:`typed` checks an array's element type where it arrives and
hands it on as one; an element then reads through ``.item()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np


if TYPE_CHECKING:
    from numpy.typing import NDArray


type Shaped[S: np.generic] = np.ndarray[tuple[int, ...], np.dtype[S]]
"""An array of ``S`` whose extents type as ints."""


def typed[S: np.generic](values: object, kind: type[S]) -> Shaped[S]:
    """Return ``values`` as an array of ``kind``, checked.

    Args:
      values: An array that numpy's stubs leave untyped.
      kind: The scalar type its elements must have.

    Returns:
      array: ``values`` itself.

    Raises:
      TypeError: ``values`` is not an array of ``kind``.

    """
    if not isinstance(values, np.ndarray) or values.dtype.type is not kind:
        raise TypeError(f"expected an array of {kind.__name__}, got {values!r:.80}")
    return cast("Shaped[S]", values)


def ints(values: NDArray[np.integer]) -> list[int]:
    """Return a one-dimensional integer array's elements as ints.

    numpy's stubs type ``tolist`` and iteration as ``Any``; ``item`` is typed.

    Args:
      values: A one-dimensional array of integers.

    Returns:
      elements: Each element, in order.

    """
    return [values.item(k) for k in range(len(values))]


def int_rows(table: Shaped[np.integer]) -> list[tuple[int, ...]]:
    """Return a two-dimensional integer array's rows as tuples of ints.

    Args:
      table: A two-dimensional array of integers.

    Returns:
      rows: Each row, in order.

    """
    rows, cols = table.shape
    return [tuple(table.item(i, j) for j in range(cols)) for i in range(rows)]
