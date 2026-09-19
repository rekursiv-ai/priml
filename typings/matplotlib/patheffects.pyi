from collections.abc import Iterable, Sequence
from typing import Any

from matplotlib.backend_bases import GraphicsContextBase, RendererBase
from matplotlib.patches import Patch
from matplotlib.path import Path
from matplotlib.transforms import Transform
from matplotlib.typing import ColorType

class AbstractPathEffect:
    def __init__(self, offset: tuple[float, float] = ...) -> None: ...
    def draw_path(
        self,
        renderer: RendererBase,
        gc: GraphicsContextBase,
        tpath: Path,
        affine: Transform,
        rgbFace: ColorType | None = ...,
    ) -> None: ...

class PathEffectRenderer(RendererBase):
    def __init__(
        self,
        path_effects: Iterable[AbstractPathEffect],
        renderer: RendererBase,
    ) -> None: ...
    def copy_with_path_effect(
        self,
        path_effects: Iterable[AbstractPathEffect],
    ) -> PathEffectRenderer: ...
    def draw_path(
        self,
        gc: GraphicsContextBase,
        tpath: Path,
        affine: Transform,
        rgbFace: ColorType | None = ...,
    ) -> None: ...
    def draw_markers(
        self,
        gc: GraphicsContextBase,
        marker_path: Path,
        marker_trans: Transform,
        path: Path,
        *args,
        **kwargs: Any,
    ) -> None: ...
    def draw_path_collection(
        self,
        gc: GraphicsContextBase,
        master_transform: Transform,
        paths: Sequence[Path],
        *args,
        **kwargs: Any,
    ) -> None: ...
    def __getattribute__(self, name: str) -> Any: ...

class Normal(AbstractPathEffect): ...

class Stroke(AbstractPathEffect):
    def __init__(self, offset: tuple[float, float] = ..., **kwargs: Any) -> None: ...
    def draw_path(
        self,
        renderer: RendererBase,
        gc: GraphicsContextBase,
        tpath: Path,
        affine: Transform,
        rgbFace: ColorType,
    ) -> None: ...

class withStroke(Stroke): ...

class SimplePatchShadow(AbstractPathEffect):
    def __init__(
        self,
        offset: tuple[float, float] = ...,
        shadow_rgbFace: ColorType | None = ...,
        alpha: float | None = ...,
        rho: float = ...,
        **kwargs: Any,
    ) -> None: ...
    def draw_path(
        self,
        renderer: RendererBase,
        gc: GraphicsContextBase,
        tpath: Path,
        affine: Transform,
        rgbFace: ColorType,
    ) -> None: ...

class withSimplePatchShadow(SimplePatchShadow): ...

class SimpleLineShadow(AbstractPathEffect):
    def __init__(
        self,
        offset: tuple[float, float] = ...,
        shadow_color: ColorType = ...,
        alpha: float = ...,
        rho: float = ...,
        **kwargs: Any,
    ) -> None: ...
    def draw_path(
        self,
        renderer: RendererBase,
        gc: GraphicsContextBase,
        tpath: Path,
        affine: Transform,
        rgbFace: ColorType,
    ) -> None: ...

class PathPatchEffect(AbstractPathEffect):
    patch: Patch
    def __init__(self, offset: tuple[float, float] = ..., **kwargs: Any) -> None: ...
    def draw_path(
        self,
        renderer: RendererBase,
        gc: GraphicsContextBase,
        tpath: Path,
        affine: Transform,
        rgbFace: ColorType,
    ) -> None: ...

class TickedStroke(AbstractPathEffect):
    def __init__(
        self,
        offset: tuple[float, float] = ...,
        spacing: float = ...,
        angle: float = ...,
        length: float = ...,
        **kwargs: Any,
    ) -> None: ...
    def draw_path(
        self,
        renderer: RendererBase,
        gc: GraphicsContextBase,
        tpath: Path,
        affine: Transform,
        rgbFace: ColorType,
    ) -> None: ...

class withTickedStroke(TickedStroke): ...
