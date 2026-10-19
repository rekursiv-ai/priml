"""Squeeze-excitation with independently injected projections and tensor layouts."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from functools import partial
from typing import Self, override

import math

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.cost import Cost, cost, reduction_cost, traffic
from priml.math.custom_types import TensorFn
from priml.model.conv import Conv2d
from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    PreservesGrid,
    TensorModule,
    infer_same_width,
    propagate_attr,
    validate_same_width,
)
from priml.model.pool import avg_pool_cost
from priml.model.swiglu import relu, sigmoid


class SqueezeExcitation(nn.Module):
    """Gate channels through mean squeeze, two projections, and an injected gate."""

    class Config(PreservesGrid, Fig["SqueezeExcitation"], kw_only=False):
        channels_in: int = -1
        """Input channel count."""

        channels_out: int = -1
        """Channel count preserved by gating."""

        _: KW_ONLY

        channels_hidden: int = -1
        """Explicit bottleneck width; the enclosing recipe owns its reduction rule."""

        fc1: Makeable[TensorModule] = field(
            default_factory=lambda: Conv2d.Config(kernel_size=1, padding=0),
        )
        """Bottleneck projection, including its weight layout and initialization."""

        fc2: Makeable[TensorModule] = field(
            default_factory=lambda: Conv2d.Config(kernel_size=1, padding=0),
        )
        """Projection back to the input channel width."""

        squeeze: TensorFn = partial(torch.mean, dim=(2, 3), keepdim=True)
        """Mean each channel to one value, in the first projection's input layout."""

        expand: TensorFn | None = None
        """View projected gates in broadcast layout; None retains the squeeze layout."""

        act: TensorFn = relu
        """Cost-bearing activation between projections."""

        gate: TensorFn = sigmoid
        """Cost-bearing gate applied after the second projection."""

        @override
        def finalize(self) -> Self:
            infer_same_width(self)
            for child, incoming, outgoing in (
                (self.fc1, self.channels_in, self.channels_hidden),
                (self.fc2, self.channels_hidden, self.channels_out),
            ):
                propagate_attr(child, "channels_in", incoming, protocol=ChannelsIn)
                propagate_attr(child, "channels_out", outgoing, protocol=ChannelsOut)
            return super().finalize()

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            seq_len: int = 1,
            input_grid: tuple[int, ...] = (),
            **kwargs: object,
        ) -> Cost:
            """Cost a per-channel mean, projections, gate, and both input-gradient paths.

            Squeeze must compute one mean per channel and expand may only view
            those gates; alternate layouts preserve this arithmetic contract.

            Args:
              batch_size: Images per invocation.
              dtype: Activation dtype.
              seq_len: Positions per channel when no spatial grid is supplied.
              input_grid: Optional spatial extents, taking precedence over seq_len.
              **kwargs: Metadata forwarded to projection costs.

            Returns:
              execution: Complete ownership, arithmetic, and logical traffic.

            """
            positions = math.prod(input_grid) if input_grid else seq_len
            gates = batch_size * self.channels_in
            elements = gates * positions
            return (
                avg_pool_cost(
                    channels=self.channels_in,
                    positions=positions,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    self.fc1,
                    seq_len=1,
                    input_grid=(1, 1),
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + cost(
                    self.act,
                    channels=batch_size * self.channels_hidden,
                    dtype=dtype,
                )
                + cost(
                    self.fc2,
                    seq_len=1,
                    input_grid=(1, 1),
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + cost(self.gate, channels=gates, dtype=dtype)
                + traffic(
                    "primal",
                    "elementwise",
                    elements=2 * elements + gates,
                    flops=elements,
                    dtype=dtype,
                )
                + traffic(
                    "adjoint",
                    "elementwise",
                    elements=8 * elements + gates,
                    flops=3 * elements,
                    dtype=dtype,
                )
                + reduction_cost(
                    input_elements=elements,
                    output_groups=gates,
                    phase="adjoint",
                    dtype=dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        validate_same_width(config)
        if config.channels_hidden < 1:
            raise ValueError("channels_hidden must be positive.")
        super().__init__()
        self.fc1 = config.fc1.make()
        self.fc2 = config.fc2.make()
        self.squeeze = config.squeeze
        self.expand = config.expand
        self.act = config.act
        self.gate = config.gate

    def reset_parameters(self) -> None:
        """Reset the two projections in their registration order."""
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        gates = self.gate(self.fc2(self.act(self.fc1(self.squeeze(x)))))
        return x * (self.expand(gates) if self.expand is not None else gates)
