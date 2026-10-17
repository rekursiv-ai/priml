"""Small forecast projections that learn corrections to the input mean."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, override

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.cost import Cost, cost, elementwise_cost, reduction_cost, traffic
from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    TensorModule,
    propagate_attr,
)
from priml.model.linear import Linear


class LowRankProjection(nn.Module):
    """Add a small learned forecast to the input mean."""

    class Config(Fig["LowRankProjection"], kw_only=False):
        channels_in: int = -1
        """History length, supplied by DLinear."""

        channels_out: int = -1
        """Forecast length, supplied by DLinear."""

        _: KW_ONLY

        channels_hidden: int = 32
        """Width between the two projections."""

        proj_in: Makeable[TensorModule] = field(default_factory=Linear.Config)
        """Compress the input window."""

        proj_out: Makeable[TensorModule] = field(
            default_factory=lambda: Linear.Config(
                bias=True,
                init_weight=nn.init.zeros_,
            ),
        )
        """Start with no correction, then learn the forecast."""

        @override
        def finalize(self) -> Self:
            for child, inputs, outputs in (
                (self.proj_in, self.channels_in, self.channels_hidden),
                (self.proj_out, self.channels_hidden, self.channels_out),
            ):
                propagate_attr(child, "channels_in", inputs, protocol=ChannelsIn)
                propagate_attr(child, "channels_out", outputs, protocol=ChannelsOut)
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count both projections, the mean, and their gradient join.

            Args:
              seq_len: Rows projected per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: Forwarded to both child costs.

            Returns:
              cost: Forward and adjoint operations and parameters of one call.

            """
            rows = seq_len * batch_size
            return (
                cost(
                    self.proj_in,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + cost(
                    self.proj_out,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + reduction_cost(
                    input_elements=rows * self.channels_in,
                    output_groups=rows,
                    dtype=dtype,
                )
                + traffic(
                    "primal",
                    "elementwise",
                    elements=3 * rows,
                    flops=rows,
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=rows * self.channels_out,
                    adjoint=0,
                    channels=self.channels_out,
                    rows=rows,
                    inputs=2,
                    adjoint_inputs=1,
                    adjoint_outputs=2,
                    dtype=dtype,
                )
                + reduction_cost(
                    input_elements=rows * self.channels_out,
                    output_groups=rows,
                    phase="adjoint",
                    dtype=dtype,
                )
                + traffic(
                    "adjoint",
                    "elementwise",
                    elements=3 * rows,
                    flops=rows,
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=0,
                    adjoint=rows * self.channels_in,
                    channels=self.channels_in,
                    rows=rows,
                    inputs=0,
                    outputs=0,
                    adjoint_inputs=2,
                    adjoint_outputs=1,
                    dtype=dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        if min(config.channels_in, config.channels_out, config.channels_hidden) <= 0:
            raise ValueError("Projection widths must be positive.")
        self.proj_in = config.proj_in.make()
        self.proj_out = config.proj_out.make()

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Predict the correction and restore the window's average."""
        return self.proj_out(self.proj_in(x)) + x.mean(dim=-1, keepdim=True)
