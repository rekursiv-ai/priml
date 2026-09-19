from typing import Any

_log = ...

def do_constrained_layout(
    fig,
    h_pad,
    w_pad,
    hspace=...,
    wspace=...,
    rect=...,
    compress=...,
) -> dict[Any, Any] | None: ...
def make_layoutgrids(fig, layoutgrids, rect=...) -> dict[Any, Any]: ...
def make_layoutgrids_gs(layoutgrids, gs): ...
def check_no_collapsed_axes(layoutgrids, fig) -> bool: ...
def compress_fixed_aspect(layoutgrids, fig): ...
def get_margin_from_padding(
    obj,
    *,
    w_pad=...,
    h_pad=...,
    hspace=...,
    wspace=...,
) -> dict[str, int]: ...
def make_layout_margins(
    layoutgrids,
    fig,
    renderer,
    *,
    w_pad=...,
    h_pad=...,
    hspace=...,
    wspace=...,
) -> None: ...
def make_margin_suptitles(
    layoutgrids,
    fig,
    renderer,
    *,
    w_pad=...,
    h_pad=...,
) -> None: ...
def match_submerged_margins(layoutgrids, fig): ...
def get_cb_parent_spans(cbax) -> tuple[range, range]: ...
def get_pos_and_bbox(ax, renderer) -> tuple[Any, Any]: ...
def reposition_axes(
    layoutgrids,
    fig,
    renderer,
    *,
    w_pad=...,
    h_pad=...,
    hspace=...,
    wspace=...,
) -> None: ...
def reposition_colorbar(layoutgrids, cbax, renderer, *, offset=...) -> None: ...
def reset_margins(layoutgrids, fig) -> None: ...
def colorbar_get_pad(layoutgrids, cax): ...
