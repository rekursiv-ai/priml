from typing import Any, Self

import numpy as np

__all__ = []
type _NumPyArrayNumeric = np.ndarray[
    tuple[int, ...],
    np.dtype[np.integer[Any] | np.floating[Any]],
]

class Mat(np.ndarray[tuple[int, ...], np.dtype[np.integer[Any] | np.floating[Any]]]):
    def __new__(cls, arr: object, **kwargs: object) -> Self: ...
    def __init__(self, arr: object, **kwargs: object) -> None: ...
    def __array_finalize__(self, obj: object) -> None: ...
