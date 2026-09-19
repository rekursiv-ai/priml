from collections.abc import Iterable
from os import PathLike
from typing import Any, overload

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.legend import Legend
from matplotlib.typing import ColorType

class Figure:
    def get_facecolor(self) -> ColorType: ...
    def set_facecolor(self, color: ColorType | None) -> None: ...
    def add_axes(
        self,
        rect: tuple[float, float, float, float],
        **kwargs: Any,
    ) -> Axes: ...
    @overload
    def legend(self) -> Legend: ...
    @overload
    def legend(
        self,
        handles: Iterable[Artist | tuple[Artist, ...]],
        labels: Iterable[str],
        **kwargs: Any,
    ) -> Legend: ...
    @overload
    def legend(
        self,
        *,
        handles: Iterable[Artist | tuple[Artist, ...]],
        **kwargs: Any,
    ) -> Legend: ...
    @overload
    def legend(self, labels: Iterable[str], **kwargs: Any) -> Legend: ...
    @overload
    def legend(self, **kwargs: Any) -> Legend: ...
    def savefig(
        self,
        fname: str | PathLike[str],
        *,
        dpi: float | str | None = ...,
        bbox_inches: str | None = ...,
        transparent: bool | None = ...,
        **kwargs: Any,
    ) -> None: ...
    def colorbar(
        self,
        mappable: ScalarMappable,
        ax: Axes | Iterable[Axes] | None = ...,
        **kwargs: Any,
    ) -> Colorbar: ...
    def tight_layout(self, **kwargs: Any) -> None: ...
    def suptitle(self, t: str, **kwargs: Any) -> Any: ...
    def subplots_adjust(
        self,
        left: float | None = ...,
        bottom: float | None = ...,
        right: float | None = ...,
        top: float | None = ...,
        wspace: float | None = ...,
        hspace: float | None = ...,
    ) -> None: ...
    def get_figwidth(self) -> float: ...
    def get_figheight(self) -> float: ...
