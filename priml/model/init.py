"""Initialization utilities."""

from __future__ import annotations

from collections.abc import Callable

import inspect
import math

from torch import Tensor, nn

from priml.model.custom_types import DepthIndex, flatten_depth_index


InitFn = Callable[[Tensor], object] | Callable[[Tensor, DepthIndex], object]


def call_init(fn: InitFn, t: Tensor, **kwargs: object) -> None:
    """Call init fn, passing kwargs only if the fn accepts them."""
    try:
        sig = inspect.signature(fn)
        accepts_kwargs = any(
            parameter.kind is parameter.VAR_KEYWORD
            for parameter in sig.parameters.values()
        )
        if not accepts_kwargs:
            supported = {
                name
                for name, parameter in sig.parameters.items()
                if parameter.kind
                in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
            }
            kwargs = {
                name: value for name, value in kwargs.items() if name in supported
            }
    except (ValueError, TypeError):
        kwargs = {}
    fn(t, **kwargs)  # pyright: ignore[reportCallIssue]  # ty: ignore[missing-argument]


def _depth_index_scale(w: Tensor, depth_index: DepthIndex) -> None:
    flattened = flatten_depth_index(depth_index)
    if flattened > 0:
        w.data /= (flattened + 1) ** 0.5


def kaiming_uniform(w: Tensor, *, depth_index: DepthIndex = ()) -> None:
    """Kaiming uniform, scaled by 1/sqrt(depth_index)."""
    nn.init.kaiming_uniform_(w, a=5**0.5)
    _depth_index_scale(w, depth_index)


def kaiming_normal(w: Tensor, *, depth_index: DepthIndex = ()) -> None:
    """Kaiming normal, scaled by 1/sqrt(depth_index)."""
    nn.init.kaiming_normal_(w, a=5**0.5)
    _depth_index_scale(w, depth_index)


def xavier_uniform(w: Tensor, *, depth_index: DepthIndex = ()) -> None:
    """Xavier uniform, scaled by 1/sqrt(depth_index)."""
    nn.init.xavier_uniform_(w)
    _depth_index_scale(w, depth_index)


def xavier_normal(w: Tensor, *, depth_index: DepthIndex = ()) -> None:
    """Xavier normal, scaled by 1/sqrt(depth_index)."""
    nn.init.xavier_normal_(w)
    _depth_index_scale(w, depth_index)


def normal(w: Tensor, *, std: float = 0.02, depth_index: DepthIndex = ()) -> None:
    """Normal init, scaled by 1/sqrt(depth_index)."""
    nn.init.normal_(w, std=std)
    _depth_index_scale(w, depth_index)


def truncated_normal(
    w: Tensor,
    *,
    std: float = 0.02,
    depth_index: DepthIndex = (),
    lower: float = -2.0,
    upper: float = 2.0,
    variance_correction: bool = False,
) -> None:
    """Truncated normal, scaled by 1/sqrt(depth_index).

    Args:
      w: Tensor to initialize in place.
      std: Standard deviation. The pre-truncation parameter by default; the
        realized value when ``variance_correction`` is set.
      depth_index: Block depth_index index; >0 divides by sqrt(depth_index + 1).
      lower: Lower truncation bound in units of ``std``.
      upper: Upper truncation bound in units of ``std``.
      variance_correction: Divide by the truncated distribution's own standard
        deviation so the realized value matches ``std``. Truncation removes
        tail mass, so the uncorrected default realizes about 0.88x the request
        at the default bounds. Matches the JAX/flax default initializer, which
        reference ports depend on for bit-parity.

    References:
      https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py
        ``jax.nn.initializers.variance_scaling`` (truncated distribution).

    """
    if variance_correction:
        if std == 0:
            nn.init.zeros_(w)
            return
        sqrt2 = 2.0**0.5
        z = (math.erf(upper / sqrt2) - math.erf(lower / sqrt2)) / 2.0
        inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
        pdf_u = inv_sqrt_2pi * math.exp(-0.5 * upper * upper)
        pdf_l = inv_sqrt_2pi * math.exp(-0.5 * lower * lower)
        # Std of N(0,1) truncated to [lower, upper].
        ratio = (pdf_u - pdf_l) / z
        std /= math.sqrt(1.0 - (upper * pdf_u - lower * pdf_l) / z - ratio * ratio)
    nn.init.trunc_normal_(w, std=std, a=lower * std, b=upper * std)
    _depth_index_scale(w, depth_index)


def unit_fan_in_uniform(w: Tensor, *, depth_index: DepthIndex = ()) -> None:
    """Uniform on ``+-sqrt(3 / fan_in)``, realizing a ``1/sqrt(fan_in)`` std.

    Depth-independent, unlike every other initializer here: ``depth_index`` is
    accepted for the :data:`InitFn` protocol and discarded.

    Args:
      w: Tensor to initialize in place.
      depth_index: Ignored.

    """
    del depth_index
    bound = 3**0.5 * w.shape[-1] ** -0.5
    nn.init.uniform_(w, -bound, bound)


def mup_output(w: Tensor, *, depth_index: DepthIndex = ()) -> None:
    """MuP output projection init: 1/fan_in, scaled by 1/sqrt(depth_index)."""
    fan_in = w.shape[1] if w.ndim >= 2 else w.shape[0]
    nn.init.normal_(w, std=1 / fan_in)
    _depth_index_scale(w, depth_index)


def dirac(w: Tensor) -> None:
    """Dirac initialization for conv weights (identity-like).

    Sets conv weights so the layer initially acts as an identity
    (or near-identity). Requires ndim >= 3.
    """
    nn.init.dirac_(w)
