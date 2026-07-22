"""Initialization utilities."""

from __future__ import annotations

from collections.abc import Callable

import inspect

from torch import Tensor, nn


InitFn = Callable[[Tensor], object] | Callable[[Tensor, int], object]


def call_init(fn: InitFn, t: Tensor, **kwargs: object) -> None:
    """Call init fn, passing kwargs only if the fn accepts them."""
    try:
        sig = inspect.signature(fn)
        supported = {
            k
            for k, v in sig.parameters.items()
            if v.kind in (v.KEYWORD_ONLY, v.VAR_KEYWORD)
        }
        kwargs = {k: v for k, v in kwargs.items() if k in supported}
    except (ValueError, TypeError):
        kwargs = {}
    fn(t, **kwargs)  # pyright: ignore[reportCallIssue]  # ty: ignore[missing-argument]


def _depth_scale(w: Tensor, depth: int) -> None:
    if depth > 0:
        w.data /= (depth + 1) ** 0.5


def kaiming_uniform(w: Tensor, *, depth: int = 1) -> None:
    """Kaiming uniform, scaled by 1/sqrt(depth)."""
    nn.init.kaiming_uniform_(w, a=5**0.5)
    _depth_scale(w, depth)


def kaiming_normal(w: Tensor, *, depth: int = 1) -> None:
    """Kaiming normal, scaled by 1/sqrt(depth)."""
    nn.init.kaiming_normal_(w, a=5**0.5)
    _depth_scale(w, depth)


def xavier_uniform(w: Tensor, *, depth: int = 1) -> None:
    """Xavier uniform, scaled by 1/sqrt(depth)."""
    nn.init.xavier_uniform_(w)
    _depth_scale(w, depth)


def xavier_normal(w: Tensor, *, depth: int = 1) -> None:
    """Xavier normal, scaled by 1/sqrt(depth)."""
    nn.init.xavier_normal_(w)
    _depth_scale(w, depth)


def normal(w: Tensor, *, std: float = 0.02, depth: int = 1) -> None:
    """Normal init, scaled by 1/sqrt(depth)."""
    nn.init.normal_(w, std=std)
    _depth_scale(w, depth)


def truncated_normal(w: Tensor, *, std: float = 0.02, depth: int = 1) -> None:
    """Truncated normal (±2 std), scaled by 1/sqrt(depth)."""
    nn.init.trunc_normal_(w, std=std, a=-2 * std, b=2 * std)
    _depth_scale(w, depth)


def mup_output(w: Tensor, *, depth: int = 1) -> None:
    """MuP output projection init: 1/fan_in, scaled by 1/sqrt(depth)."""
    fan_in = w.shape[1] if w.ndim >= 2 else w.shape[0]
    nn.init.normal_(w, std=1 / fan_in)
    _depth_scale(w, depth)


def dirac(w: Tensor) -> None:
    """Dirac initialization for conv weights (identity-like).

    Sets conv weights so the layer initially acts as an identity
    (or near-identity). Requires ndim >= 3.
    """
    nn.init.dirac_(w)
