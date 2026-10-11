"""Typed, dtype-checked reads of ``.npy`` files.

``np.load`` is typed ``Any``: what it returns depends on the bytes on disk. A
``cast`` at the call site states a dtype without checking it, so a builder that
writes ``int32`` under a reader that claims ``int64`` type-checks and runs.
``load_array`` is the one place that claim is made, and it is verified.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def load_array[T: np.generic](
    path: Path,
    dtype: type[T],
    *,
    mmap: bool = False,
) -> NDArray[T]:
    """Read one ``.npy`` array, checking it holds ``dtype``.

    Args:
      path: An ``.npy`` file.
      dtype: The element type the file must hold; an abstract type such as
        ``np.integer`` accepts any of its concrete subtypes.
      mmap: Memory-map the file read-only instead of reading it into memory.

    Returns:
      array: The stored array, typed ``NDArray[dtype]``.

    Raises:
      TypeError: The file holds something else -- an ``.npz`` archive, or an
        array of another dtype.

    """
    loaded = cast(object, np.load(path, mmap_mode="r" if mmap else None))
    if not isinstance(loaded, np.ndarray):
        raise TypeError(f"{path}: expected an array, got {type(loaded).__name__}.")
    if not np.issubdtype(loaded.dtype, dtype):
        raise TypeError(f"{path}: expected {dtype.__name__}, got {loaded.dtype}.")
    return loaded
