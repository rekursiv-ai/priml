from collections.abc import Callable, Hashable, Sequence
from typing import Any, Literal, TypeVar

import pathlib

from . import path
from ._enums import CapStyle, JoinStyle
from .artist import Artist
from .backend_bases import RendererBase
from .markers import MarkerStyle
from .transforms import Bbox, Transform

"""
Typing support for Matplotlib

This module contains Type aliases which are useful for Matplotlib and potentially
downstream libraries.

.. admonition:: Provisional status of typing

    The ``typing`` module and type stub files are considered provisional and may change
    at any time without a deprecation period.
"""
type RGBColorType = tuple[float, float, float] | str
type RGBAColorType = (
    str
    | tuple[float, float, float, float]
    | tuple[RGBColorType, float]
    | tuple[tuple[float, float, float, float], float]
)
type ColorType = RGBColorType | RGBAColorType
type RGBColourType = RGBColorType
type RGBAColourType = RGBAColorType
type ColourType = ColorType
type LineStyleType = str | tuple[float, Sequence[float]]
type DrawStyleType = Literal["default", "steps", "steps-pre", "steps-mid", "steps-post"]
type MarkEveryType = (
    int
    | tuple[int, int]
    | slice
    | list[int]
    | float
    | tuple[float, float]
    | list[bool]
    | None
)
type MarkerType = str | path.Path | MarkerStyle
type FillStyleType = Literal["full", "left", "right", "bottom", "top", "none"]
type JoinStyleType = JoinStyle | Literal["miter", "round", "bevel"]
type CapStyleType = CapStyle | Literal["butt", "projecting", "round"]
type CoordsBaseType = (
    str | Artist | Transform | Callable[[RendererBase], Bbox | Transform]
)
type CoordsType = CoordsBaseType | tuple[CoordsBaseType, CoordsBaseType]
type RcStyleType = (
    str | dict[str, Any] | pathlib.Path | Sequence[str | pathlib.Path | dict[str, Any]]
)
_HT = TypeVar("_HT", bound=Hashable)
type HashableList[_HT: Hashable] = list[_HT | HashableList[_HT]]
