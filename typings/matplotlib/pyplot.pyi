from os import PathLike
from typing import Any, Literal, overload

from matplotlib import RcParams
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from numpy.typing import NDArray

rcParams: RcParams

def switch_backend(newbackend: str) -> None: ...
def figure(
    num: int | str | Figure | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: float | None = None,
    **kwargs: Any,
) -> Figure: ...
@overload
def subplots(
    nrows: Literal[1] = 1,
    ncols: Literal[1] = 1,
    *,
    sharex: bool | str = False,
    sharey: bool | str = False,
    squeeze: Literal[True] = True,
    width_ratios: Any = None,
    height_ratios: Any = None,
    subplot_kw: dict[str, Any] | None = None,
    gridspec_kw: dict[str, Any] | None = None,
    **fig_kw: Any,
) -> tuple[Figure, Axes]: ...
@overload
def subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    sharex: bool | str = False,
    sharey: bool | str = False,
    squeeze: bool = True,
    width_ratios: Any = None,
    height_ratios: Any = None,
    subplot_kw: dict[str, Any] | None = None,
    gridspec_kw: dict[str, Any] | None = None,
    **fig_kw: Any,
) -> tuple[Figure, NDArray[Any]]: ...
def savefig(
    fname: str | PathLike[str],
    *,
    dpi: float | str = ...,
    bbox_inches: str | None = ...,
    transparent: bool | None = ...,
    **kwargs: Any,
) -> None: ...
def close(fig: Figure | str | int | None = ...) -> None: ...
def get_cmap(name: str | Colormap | None = ..., lut: int | None = ...) -> Colormap: ...
def tight_layout(**kwargs: Any) -> None: ...
