"""Output gating for attention modules."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, cast, override

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.model.attention.kvcache import KVCache
from priml.model.attention.self_attention import SelfAttention
from priml.model.cost import Cost, cost, elementwise_cost, matmul_cost
from priml.model.custom_types import (
    CachedAttention,
    ChannelsHead,
    ChannelsIn,
    ChannelsOut,
    DepthIndex,
    HasDepthIndex,
    NumHeads,
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

        inner: Makeable[TensorModule] = field(default_factory=SelfAttention.Config)
        """Wrapped attention module config."""

        bias: bool = False
        """Include bias in the gate projection."""

        depth_index: DepthIndex = ()
        """Block depth index for depth-scaled init (-1 = no scaling)."""

        @property
        def num_heads(self) -> int:
            """Return the wrapped module's attention-head count."""
            return self.inner.num_heads if isinstance(self.inner, NumHeads) else 1

        @property
        def channels_head(self) -> int:
            """Return the wrapped module's per-head channel width."""
            if isinstance(self.inner, ChannelsHead):
                return self.inner.channels_head
            return self.channels_in

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
            rows: int = 1,
            itemsize: int = 4,
            **kwargs: object,
        ) -> Cost:
            """Count projection, sigmoid, product and the two-path input gradient.

            Args:
              rows: Rows sharing each parameter; divides its gradient reduction.
              itemsize: Uniform bytes per tensor element.
              **kwargs: The open message bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            return (
                cost(self.inner, itemsize=itemsize, rows=rows, **kwargs)
                + matmul_cost(
                    channels_in=self.channels_in,
                    channels_out=self.channels_in,
                    bias=self.bias,
                    itemsize=itemsize,
                    rows=rows,
                )
                + elementwise_cost(
                    primal=5 * self.channels_in,
                    adjoint=6 * self.channels_in,
                    channels=self.channels_in,
                    inputs=4,
                    outputs=1,
                    adjoint_inputs=9,
                    adjoint_outputs=3,
                    rows=rows,
                    itemsize=itemsize,
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

    def alloc_kv_cache(
        self,
        *,
        batch: int | tuple[int, ...],
        max_seq: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KVCache:
        """Allocate the wrapped attention's cache.

        Args:
          batch: Batch size or shape tuple.
          max_seq: Maximum sequence length.
          device: Device placement (default: module device).
          dtype: Tensor dtype (default: module dtype).

        Returns:
          cache: Empty KV cache ready for generation.

        """
        inner = cast(CachedAttention[KVCache], self.inner)
        return inner.alloc_kv_cache(
            batch=batch,
            max_seq=max_seq,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(
        self,
        x: Tensor,
        **kwargs: object,
    ) -> Tensor:
        gate = torch.sigmoid(self.gate_proj(x))
        return self.inner(x, **kwargs) * gate

    def forward_cached(
        self,
        x: Tensor,
        *,
        cache: KVCache,
        **kwargs: object,
    ) -> tuple[Tensor, KVCache]:
        """Apply the gate while updating the wrapped attention's cache.

        Args:
          x: X.
          cache: Cache.
          **kwargs: Kwargs.

        Returns:
          result: The tuple[Tensor, KVCache].

        """
        gate = torch.sigmoid(self.gate_proj(x))
        inner = cast(CachedAttention[KVCache], self.inner)
        out, updated = inner.forward_cached(x, cache=cache, **kwargs)
        return out * gate, updated
