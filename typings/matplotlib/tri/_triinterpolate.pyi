from typing import Literal

from matplotlib.tri import Triangulation, TriFinder
from numpy.typing import ArrayLike

import numpy as np

class TriInterpolator:
    def __init__(
        self,
        triangulation: Triangulation,
        z: ArrayLike,
        trifinder: TriFinder | None = ...,
    ) -> None: ...
    def __call__(self, x: ArrayLike, y: ArrayLike) -> np.ma.MaskedArray: ...
    def gradient(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray]: ...

class LinearTriInterpolator(TriInterpolator): ...

class CubicTriInterpolator(TriInterpolator):
    def __init__(
        self,
        triangulation: Triangulation,
        z: ArrayLike,
        kind: Literal["min_E", "geom", "user"] = ...,
        trifinder: TriFinder | None = ...,
        dz: tuple[ArrayLike, ArrayLike] | None = ...,
    ) -> None: ...

__all__ = ("CubicTriInterpolator", "LinearTriInterpolator", "TriInterpolator")
