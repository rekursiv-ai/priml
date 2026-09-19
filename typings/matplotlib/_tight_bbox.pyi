from collections.abc import Callable
from typing import Any

def adjust_bbox(fig, bbox_inches, fixed_dpi=...) -> Callable[[], None]: ...
def process_figure_for_rasterizing(
    fig,
    bbox_inches_restore,
    fixed_dpi=...,
) -> tuple[Any, Callable[[], None]]: ...
