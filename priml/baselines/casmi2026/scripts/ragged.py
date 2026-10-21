"""Save arrays that were streamed to a temporary file without holding them in memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from pathlib import Path
    from typing import IO


def save_stream(
    stream: IO[bytes],
    path: Path,
    *,
    dtype: type[np.generic],
    width: int = 1,
) -> None:
    """Save every value written to ``stream`` as one ``.npy`` array.

    Args:
      stream: Temporary file holding contiguous values of ``dtype``, at its end.
      path: Destination ``.npy`` file.
      dtype: Element type of the streamed values.
      width: Values per row; one gives a vector, more a matrix of that many columns.

    """
    length = stream.tell() // np.dtype(dtype).itemsize
    shape = (length // width, width) if width > 1 else (length,)
    values = (
        np.memmap(stream, dtype=dtype, mode="r", shape=shape)
        if length
        else np.empty(shape, dtype=dtype)
    )
    np.save(path, values, allow_pickle=False)
