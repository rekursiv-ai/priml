"""Types the presolver's kernels are annotated with; Numba itself ignores annotations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, SupportsIndex, overload


if TYPE_CHECKING:
    from numpy.typing import NDArray

    import numpy as np


class Buf[T: (int, float)](Protocol):
    """A one-dimensional array as the kernels use it.

    Indexing with an integer yields a Python scalar, which numpy's own stubs give
    only for literal ints; the kernels index arrays by values read from others.
    """

    @property
    def size(self) -> int:
        """Number of elements."""
        ...

    @overload
    def __getitem__(self, index: SupportsIndex, /) -> T: ...
    @overload
    def __getitem__(
        self,
        index: slice[int | None, int | None, int | None],
        /,
    ) -> Self: ...
    @overload
    def __setitem__(self, index: SupportsIndex, value: T, /) -> None: ...
    @overload
    def __setitem__(
        self,
        index: slice[int | None, int | None, int | None],
        value: T | Buf[T],
        /,
    ) -> None: ...
    def __len__(self) -> int: ...
    def __array__(self) -> NDArray[np.generic]: ...
    def copy(self) -> Self:
        """Copy the elements into a new array."""
        ...

    def astype[D: np.generic](self, dtype: type[D], /) -> NDArray[D]:
        """Copy the elements, converted to ``dtype``."""
        ...


type FBuf = Buf[float]
"""A float64 array."""

type IBuf = Buf[int]
"""An integer array: int32 indices and sizes, or uint8 tags."""
