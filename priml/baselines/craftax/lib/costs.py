"""Costs the port's configs share: torch's activations, adds, gathers, heads, copies.

priml prices a slot's function through the ``cost`` that ``@set_cost``
attaches to it. The port's configs hold torch's own functions --
``functional.silu``, ``functional.gelu``, ``torch.tanh`` -- because their
goldens print them so, and torch's functions carry no cost; their counts live
here. ``silu`` counts as :func:`priml.model.swiglu.silu` does.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from torch.nn import functional

import torch

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    map_cost,
    matmul_cost,
    resolve_dtype,
    traffic,
)
from priml.model.embedding import Embedding


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from priml.math.custom_types import TensorFn


_TORCH_ACTIVATIONS: Final[Mapping[object, Callable[..., Cost]]] = MappingProxyType(
    {
        # ``x * sigmoid(x)``: the sigmoid's exp, add, divide and negate, then the
        # multiply; five back from the saved sigmoid.
        functional.silu: map_cost(primal=5, adjoint=5),
        # ``x / 2 * (1 + erf(x / sqrt(2)))``: scale, erf, add, halve, multiply.
        # Back, ``g * (cdf + x * pdf)``: the cdf's four, the pdf's square,
        # scale, exp and normalizer, then the product, the sum and ``g``.
        functional.gelu: map_cost(primal=5, adjoint=11),
        # One tanh; back ``g * (1 - t**2)`` on the saved output.
        torch.tanh: map_cost(primal=1, adjoint=3),
    },
)


def activation_cost(
    activation: TensorFn,
    *,
    elements: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost one activation over ``elements`` values, forward and back.

    Args:
      activation: Torch's ``silu``, ``gelu`` or ``tanh``, or any function
        priml's :func:`~priml.cost.cost` prices.
      elements: Values it maps in the invocation.
      dtype: Activation dtype; ``None`` is torch's default.

    Returns:
      cost: The map's whole-invocation cost.

    Raises:
      TypeError: ``activation`` has no cost here or of its own.

    """
    priced = _TORCH_ACTIVATIONS.get(activation)
    if priced is None:
        return cost(activation, channels=elements, dtype=dtype)
    return priced(channels=elements, dtype=dtype)


def residual_cost(*, channels: int, rows: int, dtype: torch.dtype | None) -> Cost:
    """Cost a residual add: one add forward, one gradient accumulation back.

    Args:
      channels: Width of the stream.
      rows: Rows of the stream in the invocation.
      dtype: Activation dtype; ``None`` is torch's default.

    Returns:
      cost: The add's whole-invocation cost, as priml's blocks count theirs.

    """
    return elementwise_cost(
        primal=rows * channels,
        adjoint=rows * channels,
        channels=channels,
        rows=rows,
        inputs=2,
        dtype=dtype,
    )


def broadcast_add_cost(*, params: int, rows: int, dtype: torch.dtype | None) -> Cost:
    """Cost adding a learned table to each of ``rows`` copies of a stream.

    The add passes its gradient to the stream unchanged, and the table's
    gradient sums the ``rows`` copies.

    Args:
      params: Values in the table, and in each copy it is added to.
      rows: Copies the table is added to.
      dtype: Activation dtype; ``None`` is torch's default.

    Returns:
      cost: The adds, the table's gradient reduction, and its ownership.

    """
    return elementwise_cost(
        primal=rows * params,
        adjoint=0,
        channels=params,
        params=params,
        rows=rows,
        inputs=1,
        dtype=dtype,
    )


def gather_cost(
    *,
    rows: int,
    width: int,
    source: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost gathering ``rows`` rows of ``width`` from a ``source``-row activation.

    Priced as priml's :class:`~priml.model.embedding.Embedding` prices a
    lookup, including the scatter-add adjoint into a zeroed gradient, but owning
    nothing: the rows are an activation's, not a table's.

    Args:
      rows: Rows gathered.
      width: Values per row.
      source: Rows of the tensor gathered from.
      dtype: Activation dtype; ``None`` is torch's default.

    Returns:
      cost: The gather's and its adjoint's whole-invocation cost.

    """
    lookup = Embedding.Config(channels_in=source, channels_out=width)
    return lookup.cost(seq_len=1, batch_size=rows, dtype=dtype).tile(1, copies=0)


def weight_gradient_only(projection: Cost) -> Cost:
    """Return a projection's cost when its input takes no gradient.

    Back, the product forms the weight's gradient alone: one product the
    primal's size, reading the output's gradient and the input and writing
    the weight's. A bias's gradient reduction is unchanged.

    Args:
      projection: The projection's cost with both gradients.

    Returns:
      cost: The same invocation without the input's gradient.

    """
    both = projection.only("adjoint").only("matmul").cells.keys()
    kept = Cost(
        cells={
            key: value for key, value in projection.cells.items() if key not in both
        },
        params=projection.params,
        params_active=projection.params_active,
        bytes_state=projection.bytes_state,
    )
    return kept + projection.only("primal").only("matmul").relabel("adjoint")


def logits_cost(
    *,
    channels_in: int,
    channels_out: int,
    rows: int,
    weight: bool,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost a bias-free head and its ``.float()`` copy, which float32 skips.

    Args:
      channels_in: Width of the rows the head reads.
      channels_out: Logits per row.
      rows: Rows scored.
      weight: Whether the head owns its matrix; False for a tied table.
      dtype: Activation dtype; ``None`` is torch's default.

    Returns:
      cost: The product's and the copy's whole-invocation cost.

    """
    head = matmul_cost(
        channels_in=channels_in,
        channels_out=channels_out,
        rows=rows,
        weight=weight,
        dtype=dtype,
    )
    if resolve_dtype(dtype) == torch.float32:
        return head
    return head + cast_cost(elements=rows * channels_out, dtype=dtype)


def cast_cost(*, elements: int, dtype: torch.dtype | None) -> Cost:
    """Cost a dtype cast: each value read and written, and its gradient cast back.

    Args:
      elements: Values cast.
      dtype: Activation dtype; ``None`` is torch's default.

    Returns:
      cost: The two copies' traffic, which do no arithmetic.

    """
    return traffic(
        "primal",
        "elementwise",
        elements=2 * elements,
        dtype=dtype,
    ) + traffic("adjoint", "elementwise", elements=2 * elements, dtype=dtype)


def concat_cost(*, elements: int, dtype: torch.dtype | None) -> Cost:
    """Cost a concatenation: each input read and written once; its gradient is views.

    Args:
      elements: Values in the output.
      dtype: Activation dtype; ``None`` is torch's default.

    Returns:
      cost: The copy's traffic, which does no arithmetic.

    """
    return traffic("primal", "elementwise", elements=2 * elements, dtype=dtype)
