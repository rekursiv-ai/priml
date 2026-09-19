from collections.abc import Generator
from typing import Any

import dataclasses

from .ft2font import FT2Font

"""
Low-level text helper utilities.
"""

@dataclasses.dataclass(frozen=True)
class LayoutItem:
    ft_object: FT2Font
    char: str
    glyph_idx: int
    x: float
    prev_kern: float

def warn_on_missing_glyph(codepoint, fontnames) -> None: ...
def layout(string, font, *, kern_mode=...) -> Generator[LayoutItem, Any]: ...
