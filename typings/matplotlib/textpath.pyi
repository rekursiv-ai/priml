from typing import Literal

from matplotlib.font_manager import FontProperties
from matplotlib.ft2font import FT2Font
from matplotlib.mathtext import MathTextParser, VectorParse
from matplotlib.path import Path

import numpy as np

class TextToPath:
    FONT_SCALE: float
    DPI: float
    mathtext_parser: MathTextParser[VectorParse]
    def __init__(self) -> None: ...
    def get_text_width_height_descent(
        self,
        s: str,
        prop: FontProperties,
        ismath: bool | Literal["TeX"],
    ) -> tuple[float, float, float]: ...
    def get_text_path(
        self,
        prop: FontProperties,
        s: str,
        ismath: bool | Literal["TeX"] = ...,
    ) -> list[np.ndarray]: ...
    def get_glyphs_with_font(
        self,
        font: FT2Font,
        s: str,
        glyph_map: dict[str, tuple[np.ndarray, np.ndarray]] | None = ...,
        return_new_glyphs_only: bool = ...,
    ) -> tuple[
        list[tuple[str, float, float, float]],
        dict[str, tuple[np.ndarray, np.ndarray]],
        list[tuple[list[tuple[float, float]], list[int]]],
    ]: ...
    def get_glyphs_mathtext(
        self,
        prop: FontProperties,
        s: str,
        glyph_map: dict[str, tuple[np.ndarray, np.ndarray]] | None = ...,
        return_new_glyphs_only: bool = ...,
    ) -> tuple[
        list[tuple[str, float, float, float]],
        dict[str, tuple[np.ndarray, np.ndarray]],
        list[tuple[list[tuple[float, float]], list[int]]],
    ]: ...
    def get_glyphs_tex(
        self,
        prop: FontProperties,
        s: str,
        glyph_map: dict[str, tuple[np.ndarray, np.ndarray]] | None = ...,
        return_new_glyphs_only: bool = ...,
    ) -> tuple[
        list[tuple[str, float, float, float]],
        dict[str, tuple[np.ndarray, np.ndarray]],
        list[tuple[list[tuple[float, float]], list[int]]],
    ]: ...

text_to_path: TextToPath

class TextPath(Path):
    def __init__(
        self,
        xy: tuple[float, float],
        s: str,
        size: float | None = ...,
        prop: FontProperties | None = ...,
        _interpolation_steps: int = ...,
        usetex: bool = ...,
    ) -> None: ...
    def set_size(self, size: float | None) -> None: ...
    def get_size(self) -> float | None: ...
    @property
    def vertices(self) -> np.ndarray: ...
    @property
    def codes(self) -> np.ndarray: ...
