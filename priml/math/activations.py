"""Activations torch does not ship, as plain ``Tensor -> Tensor`` functions.

Only what is missing: ``silu``, ``gelu``, ``relu`` and friends live in
``torch.nn.functional`` and are passed directly into an
:data:`~priml.model.custom_types.ActivationFn` slot. A function here earns
its place by being absent there.
"""

from __future__ import annotations

from torch import Tensor, nn


__all__ = ["relu_squared"]


def relu_squared(x: Tensor) -> Tensor:
    """``relu(x) ** 2`` -- a rectifier whose gradient is itself a rectifier.

    Not ``x ** 2``: squaring alone is even, so it discards the sign and is not
    monotone. The rectifier keeps the negative half at exactly zero, and the
    square applies above it.

    The derivative is ``2 * relu(x)``, which is continuous at the origin where
    ``relu``'s (0 below, 1 above) jumps -- so this is C¹ while keeping ReLU's
    exact zero-sparsity, which a smooth-everywhere activation (GELU, SiLU) does
    not. Growing quadratically rather than linearly also amplifies large
    activations relative to small ones, which is the multiplicative effect a
    gate otherwise buys with a second matrix.

    Args:
      x: Pre-activation values, any shape.

    Returns:
      activated: ``max(0, x) ** 2``, elementwise.

    References:
      https://arxiv.org/abs/2109.08668
        So et al. 2021, "Primer: Searching for Efficient Transformer for
        Language Modeling."

    """
    return nn.functional.relu(x).square()
