from typing import IO, Generic, Literal, TypeVar, overload

import os

from matplotlib.font_manager import FontProperties
from matplotlib.typing import ColorType

from ._mathtext import (
    RasterParse as RasterParse,
    VectorParse as VectorParse,
)

_ParseType = TypeVar("_ParseType", RasterParse, VectorParse)

class MathTextParser(Generic[_ParseType]):
    @overload
    def __init__(
        self: MathTextParser[VectorParse],
        output: Literal["path"],
    ) -> None: ...
    @overload
    def __init__(
        self: MathTextParser[RasterParse],
        output: Literal["agg", "raster", "macosx"],
    ) -> None: ...
    def parse(
        self,
        s: str,
        dpi: float = ...,
        prop: FontProperties | None = ...,
        *,
        antialiased: bool | None = ...,
    ) -> _ParseType: ...

def math_to_image(
    s: str,
    filename_or_obj: str | os.PathLike | IO,
    prop: FontProperties | None = ...,
    dpi: float | None = ...,
    format: str | None = ...,
    *,
    color: ColorType | None = ...,
) -> float: ...
