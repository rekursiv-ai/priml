"""Linear kernels with explicit activation-dtype parameter conversion."""

from __future__ import annotations

from torch import Tensor, nn


def cast_parameters_to_input(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Cast linear parameters to the input dtype before applying the projection."""
    weight = weight.to(dtype=input.dtype)
    if bias is not None:
        bias = bias.to(dtype=input.dtype)
    return nn.functional.linear(input, weight, bias)
