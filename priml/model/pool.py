"""Costs of pooling: the windowed max and the global average.

The pooling layers are torch's; what priml adds is their analytical cost, for
the networks that cost themselves layer by layer -- beside ``conv_cost`` in
``model/conv.py``.
"""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import Self, override

import math

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.cost import Cost, reduction_cost, traffic
from priml.model.custom_types import infer_same_width, validate_same_width


__all__ = ["AvgPool2d", "avg_pool_cost", "max_pool_cost"]


class AvgPool2d(nn.AvgPool2d):
    """Apply native window averaging with clipped geometry and overlap cost."""

    class Config(Fig["AvgPool2d"], kw_only=False):
        channels_in: int = -1
        """Number of input planes, or supplied on the cost bus."""

        channels_out: int = -1
        """Number of output planes, preserved by pooling."""

        _: KW_ONLY

        kernel_size: int | tuple[int, int] = 2
        """Spatial averaging window."""

        stride: int | tuple[int, int] | None = None
        """Window step; None uses the kernel extent."""

        padding: int | tuple[int, int] = 0
        """Implicit zero padding on both sides of each spatial axis."""

        ceil_mode: bool = False
        """Include a partial final window beginning inside the input."""

        count_include_pad: bool = True
        """Include implicit padding in the average divisor."""

        divisor_override: int | None = None
        """Explicit divisor instead of each window's element count."""

        @override
        def finalize(self) -> Self:
            infer_same_width(self)
            return super().finalize()

        def output_grid(self, input_grid: tuple[int, ...]) -> tuple[int, ...]:
            """Return the two spatial extents, including Torch's ceil correction.

            Args:
              input_grid: Input height and width.

            Returns:
              output: Output height and width.

            Raises:
              ValueError: The grid does not have two spatial axes.

            """
            if len(input_grid) != 2:
                raise ValueError("Average pooling requires two spatial axes.")
            output: list[int] = []
            for size, taps, step, pad in zip(
                input_grid,
                _pair(self.kernel_size),
                _pair(self.kernel_size if self.stride is None else self.stride),
                _pair(self.padding),
                strict=True,
            ):
                count = (
                    size + 2 * pad - taps + (step - 1 if self.ceil_mode else 0)
                ) // step + 1
                if self.ceil_mode and (count - 1) * step >= size + pad:
                    count -= 1
                output.append(count)
            return tuple(output)

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            channels: int = -1,
            **kwargs: object,
        ) -> Cost:
            """Count actual source visits, divisors, and overlapping input gradients.

            Padding changes divisors, never source reads. The adjoint scales
            each output gradient once, then reduces its contributions per input.

            Args:
              input_grid: Input height and width.
              batch_size: Images per invocation.
              dtype: Activation dtype.
              channels: Plane count when not configured on this leaf.
              **kwargs: Unused outer metadata.

            Returns:
              execution: Logical window traffic and complete arithmetic.

            """
            del kwargs
            grid = self.output_grid(input_grid)
            stats = [
                _window_counts(size, outputs=count, taps=taps, step=step, pad=pad)
                for size, count, taps, step, pad in zip(
                    input_grid,
                    grid,
                    _pair(self.kernel_size),
                    _pair(self.kernel_size if self.stride is None else self.stride),
                    _pair(self.padding),
                    strict=True,
                )
            ]
            planes = batch_size * (
                self.channels_in if self.channels_in != -1 else channels
            )
            visits = planes * math.prod(part[0] for part in stats)
            covered = planes * math.prod(part[1] for part in stats)
            outputs, inputs = planes * math.prod(grid), planes * math.prod(input_grid)
            return (
                reduction_cost(
                    input_elements=visits,
                    output_groups=outputs,
                    dtype=dtype,
                )
                + traffic(
                    "primal",
                    "elementwise",
                    elements=2 * outputs,
                    flops=outputs,
                    dtype=dtype,
                )
                + traffic(
                    "adjoint",
                    "elementwise",
                    elements=2 * outputs,
                    flops=outputs,
                    dtype=dtype,
                )
                + traffic(
                    "adjoint",
                    "reduction",
                    elements=visits + inputs,
                    flops=visits - covered,
                    dtype=dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        validate_same_width(config)
        super().__init__(
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            ceil_mode=config.ceil_mode,
            count_include_pad=config.count_include_pad,
            divisor_override=config.divisor_override,
        )

    def reset_parameters(self) -> None:
        """Retain the stateless averaging operation."""

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return super().forward(input)


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


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else value


def _window_counts(
    size: int,
    *,
    outputs: int,
    taps: int,
    step: int,
    pad: int,
) -> tuple[int, int]:
    visits = covered = previous_end = 0
    for position in range(outputs):
        start = max(0, position * step - pad)
        end = min(size, position * step - pad + taps)
        visits += max(0, end - start)
        covered += max(0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    return visits, covered
