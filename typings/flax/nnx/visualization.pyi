import typing as tp

from _typeshed import Incomplete
from treescope import (
    renderers as renderers,
    rendering_parts,
)

in_ipython: Incomplete

def display(*args) -> None: ...
def render_object_constructor(
    object_type: type[tp.Any],
    attributes: tp.Mapping[str, tp.Any],
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
    roundtrippable: bool = False,
    color: str | None = None,
    first_line_annotation: rendering_parts.RenderableTreePart | None = None,
) -> rendering_parts.Rendering: ...
