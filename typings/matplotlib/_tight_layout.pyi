from typing import Any

def get_subplotspec_list(axes_list, grid_spec=...) -> list[Any]: ...
def get_tight_layout_figure(
    fig,
    axes_list,
    subplotspec_list,
    renderer,
    pad=...,
    h_pad=...,
    w_pad=...,
    rect=...,
) -> dict[Any, Any] | dict[str, Any] | None: ...
