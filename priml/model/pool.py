"""Costs of pooling: the windowed max and the global average.

The pooling layers are torch's; what priml adds is their analytical cost, for
the networks that cost themselves layer by layer -- beside ``conv_cost`` in
``model/conv.py``.
"""

from __future__ import annotations

import math

import torch

from priml.cost import Cost, traffic


__all__ = ["avg_pool_cost", "max_pool_cost"]


def max_pool_cost(
    *,
    channels: int,
    kernel_size: int | tuple[int, ...],
    rows: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost a 2-d max pool: compares forward, one gradient routed to the argmax.

    The forward reads each window and writes the value and its saved argmax,
    one ``int64`` per pooled element; the adjoint reads both and scatters one
    gradient back to the argmax, writing the dense input gradient.

    Args:
      channels: Pooled channels.
      kernel_size: Window extent, scalar or per axis.
      rows: Pooled positions across the batch (batch times output grid).
      dtype: Value and gradient dtype; ``None`` is torch's default.

    Returns:
      cost: Whole-invocation FLOPs and logical bytes; owns nothing.

    """
    window = math.prod(
        kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2,
    )
    elements = channels * rows
    return (
        traffic(
            "primal",
            "reduction",
            elements=elements * (window + 1),
            flops=elements * (window - 1),
            dtype=dtype,
        )
        + traffic("primal", "reduction", elements=elements, dtype=torch.int64)
        + traffic(
            "adjoint",
            "selection",
            elements=elements * (window + 1),
            flops=elements,
            dtype=dtype,
        )
        + traffic("adjoint", "selection", elements=elements, dtype=torch.int64)
    )


def avg_pool_cost(
    *,
    channels: int,
    positions: int,
    batch_size: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost a global average pool: a sum per channel, one scale, a spread back.

    Args:
      channels: Pooled channels.
      positions: Spatial positions averaged per channel.
      batch_size: Images in this invocation.
      dtype: Value and gradient dtype; ``None`` is torch's default.

    Returns:
      cost: Whole-invocation FLOPs and logical bytes; owns nothing.

    """
    rows = batch_size * positions
    pooled = channels * batch_size
    return (
        traffic(
            "primal",
            "reduction",
            elements=channels * (rows + batch_size),
            flops=pooled * (positions - 1),
            dtype=dtype,
        )
        + traffic(
            "primal",
            "elementwise",
            elements=2 * pooled,
            flops=pooled,
            dtype=dtype,
        )
        + traffic(
            "adjoint",
            "elementwise",
            elements=channels * (rows + batch_size),
            flops=channels * rows,
            dtype=dtype,
        )
    )
