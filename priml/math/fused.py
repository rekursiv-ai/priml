"""Floating-point primitives with explicit fused rounding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch import Tensor


@torch.no_grad()
def fma32(x: Tensor, *, y: Tensor, addend: Tensor) -> Tensor:
    """Return a binary32 product plus addend with one rounding.

    This preprocessing operation does not record an autograd graph.

    Args:
        x: First binary32 multiplicand.
        y: Broadcastable binary32 second multiplicand.
        addend: Broadcastable binary32 addend.

    Returns:
        result: Correctly rounded binary32 fused multiply-add.

    Raises:
        TypeError: An operand has a different dtype.

    References:
        https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf
        Ogita, Rump and Oishi (2005), Algorithm 3.1 (TwoSum).

    """
    if any(operand.dtype != torch.float32 for operand in (x, y, addend)):
        raise TypeError("Fused binary32 arithmetic requires binary32 operands.")
    product = x.double() * y.double()
    summand = addend.double()
    total, residual = _two_sum(product, y=summand)
    rounded = total.float()
    rounded64 = rounded.double()
    bits = rounded.view(torch.int32)
    direction = torch.sign(total - rounded64)
    neighbor_bits = bits + direction.to(torch.int32) * (1 | (bits >> 31))
    neighbor = neighbor_bits.view(torch.float32)
    midpoint = (rounded64 + neighbor.double()) * 0.5
    overflow_midpoint = torch.copysign(
        torch.full_like(total, float((1 << 128) - (1 << 103))),
        total,
    )
    midpoint = torch.where(rounded.isinf(), overflow_midpoint, midpoint)
    crossed = torch.sign(residual) == direction
    return torch.where((total == midpoint) & crossed, neighbor, rounded)


def _two_sum(x: Tensor, *, y: Tensor) -> tuple[Tensor, Tensor]:
    total = x + y
    virtual = total - x
    residual = (x - (total - virtual)) + (y - virtual)
    return total, residual
