"""Output gating for attention modules."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, cast, override

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    matmul_cost,
)
from priml.model.attention.attention import Attention
from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    DepthIndex,
    HasDepthIndex,
    HeadGeometry,
    LayerCache,
    TensorModule,
    infer_same_width,
    propagate_attr,
)
from priml.model.linear import Linear


class OutputGate(nn.Module):
    """Wrap an attention module with output gating.

    Computes ``gate = gate_proj(x)`` before delegating to ``inner``,
    then applies ``out * sigmoid(gate)`` to the attention output.
    """

    class Config(Fig["OutputGate"], kw_only=False):
        channels_in: int = -1
        """Input channel width."""

        channels_out: int = -1
        """Output channel width."""

        _: KW_ONLY

        inner: Makeable[TensorModule] = field(default_factory=Attention.Config)
        """Wrapped attention module config."""

        bias: bool = False
        """Include bias in the gate projection."""

        depth_index: DepthIndex = ()
        """Block depth index for depth-scaled init (-1 = no scaling)."""

        @property
        def num_heads(self) -> int:
            """Return the wrapped module's attention-head count."""
            return cast(HeadGeometry, self.inner).num_heads

        @property
        def channels_head(self) -> int:
            """Return the wrapped module's per-head channel width."""
            return cast(HeadGeometry, self.inner).channels_head

        @override
        def finalize(self) -> Self:
            infer_same_width(self)
            propagate_attr(
                self.inner,
                "channels_in",
                self.channels_in,
                protocol=ChannelsIn,
            )
            propagate_attr(
                self.inner,
                "channels_out",
                self.channels_in,
                protocol=ChannelsOut,
            )
            propagate_attr(
                self.inner,
                "depth_index",
                self.depth_index,
                protocol=HasDepthIndex,
            )
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count projection, sigmoid, product and the two-path input gradient.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation cost over ``seq_len`` and ``batch_size``.

            """
            rows = seq_len * batch_size
            return (
                cost(
                    self.inner,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + matmul_cost(
                    channels_in=self.channels_in,
                    channels_out=self.channels_in,
                    bias=self.bias,
                    dtype=dtype,
                    rows=rows,
                )
                + elementwise_cost(
                    primal=5 * self.channels_in * rows,
                    adjoint=6 * self.channels_in * rows,
                    channels=self.channels_in,
                    inputs=4,
                    outputs=1,
                    adjoint_inputs=9,
                    adjoint_outputs=3,
                    rows=rows,
                    dtype=dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.inner = config.inner.make()
        self.gate_proj = Linear.Config(
            channels_in=config.channels_in,
            channels_out=config.channels_in,
            bias=config.bias,
            depth_index=config.depth_index,
        ).make()

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        self.inner.reset_parameters()
        self.gate_proj.reset_parameters()

    @override
    def forward(
        self,
        x: Tensor,
        *,
        cache: LayerCache | None = None,
        **kwargs: object,
    ) -> Tensor:
        # The wrapped attention owns the cache slot; the gate only forwards it.
        gate = torch.sigmoid(self.gate_proj(x))
        return self.inner(x, cache=cache, **kwargs) * gate
