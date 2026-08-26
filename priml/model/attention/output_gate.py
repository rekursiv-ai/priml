"""Output gating for attention modules."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Protocol, Self, cast, override

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.model.attention.kvcache import KVCache
from priml.model.attention.self_attention import SelfAttention
from priml.model.custom_types import (
    ChannelsHead,
    ChannelsIn,
    ChannelsOut,
    DepthIndex,
    HasDepthIndex,
    NumHeads,
    propagate_attr,
)
from priml.model.linear import Linear


class CachedAttention(Protocol):
    """An attention module with an explicit cached path."""

    def alloc_kv_cache(
        self,
        *,
        batch: int | tuple[int, ...],
        max_seq: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KVCache: ...

    def forward_cached(
        self,
        x: Tensor,
        *,
        cache: KVCache,
        **kwargs: object,
    ) -> tuple[Tensor, KVCache]: ...


class OutputGate(nn.Module):
    """Wrap an attention module with output gating.

    Computes ``gate = gate_proj(x)`` before delegating to ``inner``,
    then applies ``out * sigmoid(gate)`` to the attention output.
    """

    class Config(Fig["OutputGate"], kw_only=False):
        channels_in: int = -1
        """Model width for the gate projection."""

        channels_out: int = -1
        """Number of output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        inner: Makeable[nn.Module] = field(default_factory=SelfAttention.Config)
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
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            if self.channels_in != self.channels_out:
                raise ValueError(
                    f"channels_in={self.channels_in} must equal "
                    f"channels_out={self.channels_out} for OutputGate."
                )
            propagate_attr(
                self.inner,
                "channels_in",
                self.channels_in,
                protocol=ChannelsIn,
            )
            propagate_attr(
                self.inner,
                "channels_out",
                self.channels_out,
                protocol=ChannelsOut,
            )
            propagate_attr(
                self.inner,
                "depth_index",
                self.depth_index,
                protocol=HasDepthIndex,
            )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if (
            -1 not in (config.channels_in, config.channels_out)
            and config.channels_in != config.channels_out
        ):
            raise ValueError(
                f"channels_in={config.channels_in} must equal "
                f"channels_out={config.channels_out} for OutputGate."
            )
        super().__init__()
        self.inner = config.inner.make()
        self.gate_proj = Linear.Config(
            channels_in=config.channels_in,
            channels_out=config.channels_in,
            bias=config.bias,
            depth_index=config.depth_index,
        ).make()

    def reset_parameters(self) -> None:
        if hasattr(self.inner, "reset_parameters"):
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
        """Allocate the wrapped attention's cache."""
        inner = cast(CachedAttention, self.inner)
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
        return cast(Tensor, self.inner(x, **kwargs)) * gate

    def forward_cached(
        self,
        x: Tensor,
        *,
        cache: KVCache,
        **kwargs: object,
    ) -> tuple[Tensor, KVCache]:
        """Apply the gate while updating the wrapped attention's cache."""
        gate = torch.sigmoid(self.gate_proj(x))
        inner = cast(CachedAttention, self.inner)
        out, updated = inner.forward_cached(x, cache=cache, **kwargs)
        return out * gate, updated
