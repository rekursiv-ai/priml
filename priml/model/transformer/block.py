"""Transformer block."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from functools import partial
from typing import Protocol, Self, cast, override

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

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
    Shardable,
    TensorModule,
    propagate_attr,
)
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU


class CachedAttention(Protocol):
    """An attention sublayer with an explicit cached path."""

    def forward_cached(
        self,
        x: Tensor,
        *,
        cache: KVCache,
        **kwargs: object,
    ) -> tuple[Tensor, KVCache]: ...


class TransformerBlock(nn.Module):
    """Transformer block with attention and FFN.

    Module names match sic convention for checkpoint compatibility:
    attn, ffn, norm1 (post-attn), norm2 (post-ffn).
    """

    class Config(Fig["TransformerBlock"], kw_only=False):
        channels_in: int = -1
        """Number of input channels."""

        _: KW_ONLY

        attn: Makeable[nn.Module] = field(default_factory=SelfAttention.Config)
        """Attention module config."""

        ffn: Makeable[TensorModule] = field(default_factory=SwiGLU.Config)
        """Feed-forward module config."""

        norm1: Makeable[TensorModule] = field(default_factory=RMSNorm.Config)
        """Normalization applied around attention."""

        norm2: Makeable[TensorModule] = field(default_factory=RMSNorm.Config)
        """Normalization applied around the feed-forward."""

        prenorm: bool = True
        """Apply norm before (True) or after (False) each sublayer."""

        checkpoint: bool = False
        """Use activation checkpointing to trade compute for memory."""

        depth_index: DepthIndex = ()
        """Block depth index for depth-scaled init (-1 = no scaling)."""

        channels_out: int = -1
        """Number of output channels (-1 to infer from channels_in)."""

        @property
        def num_heads(self) -> int:
            """Return the attention sublayer's head count."""
            return self.attn.num_heads if isinstance(self.attn, NumHeads) else 1

        @property
        def channels_head(self) -> int:
            """Return the attention sublayer's per-head channel width."""
            if isinstance(self.attn, ChannelsHead):
                return self.attn.channels_head
            return self.channels_in

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            c = self.channels_in
            for cfg in (self.attn, self.ffn, self.norm1, self.norm2):
                propagate_attr(cfg, "channels_in", c, protocol=ChannelsIn)
                propagate_attr(cfg, "channels_out", c, protocol=ChannelsOut)
                propagate_attr(
                    cfg, "depth_index", self.depth_index, protocol=HasDepthIndex
                )
            # Tensor parallelism: the FFN shards over the tp dim (its block-
            # internal style handles the fused-gate split alignment). The
            # attention block's children self-declare their styles.
            if isinstance(self.ffn, Shardable):
                self.ffn.shard = "colwise"
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if (
            -1 not in (config.channels_in, config.channels_out)
            and config.channels_in != config.channels_out
        ):
            raise ValueError(
                f"channels_in={config.channels_in} must equal "
                f"channels_out={config.channels_out} for TransformerBlock."
            )
        super().__init__()
        self.prenorm = config.prenorm
        self.checkpoint = config.checkpoint
        self.depth_index = config.depth_index
        self.attn = config.attn.make()
        self.ffn = config.ffn.make()
        self.norm1 = config.norm1.make()
        self.norm2 = config.norm2.make()

    def reset_parameters(self) -> None:
        for m in (self.attn, self.ffn, self.norm1, self.norm2):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    @override
    def forward(
        self,
        x: Tensor,
        **kwargs: object,
    ) -> Tensor:
        # Gate on ``torch.is_grad_enabled()``: activation checkpointing only saves
        # memory by recomputing in backward, so it is pointless with grad off
        # (eval / ``torch.no_grad`` / ``torch.inference_mode``) -- and wrapping a
        # block in ``torch.utils.checkpoint`` under ``inference_mode`` can deadlock
        # a multi-rank eval. ``is_grad_enabled()`` is the precise condition (a
        # backward will run); it subsumes the older ``x.requires_grad`` check and
        # also covers a ``requires_grad`` input inside a ``no_grad`` region.
        if self.checkpoint and torch.is_grad_enabled():
            return torch_checkpoint(
                partial(self._forward, **kwargs),
                x,
                use_reentrant=False,
            )
        return self._forward(x, **kwargs)

    def forward_cached(
        self,
        x: Tensor,
        *,
        cache: KVCache,
        **kwargs: object,
    ) -> tuple[Tensor, KVCache]:
        """Run the block while updating its attention cache."""
        attention = cast(CachedAttention, self.attn)
        if self.prenorm:
            attn_out, cache = attention.forward_cached(
                self.norm1(x, **kwargs),
                cache=cache,
                **kwargs,
            )
            x = x + attn_out
            x = x + self.ffn(self.norm2(x, **kwargs), **kwargs)
        else:
            attn_out, cache = attention.forward_cached(x, cache=cache, **kwargs)
            x = self.norm1(x + attn_out, **kwargs)
            x = self.norm2(x + self.ffn(x, **kwargs), **kwargs)
        return x, cache

    def _forward(
        self,
        x: Tensor,
        **kwargs: object,
    ) -> Tensor:
        if self.prenorm:
            attn_out = cast(Tensor, self.attn(self.norm1(x, **kwargs), **kwargs))
            x = x + attn_out
            x = x + self.ffn(self.norm2(x, **kwargs), **kwargs)
        else:
            attn_out = cast(Tensor, self.attn(x, **kwargs))
            x = self.norm1(x + attn_out, **kwargs)
            x = self.norm2(x + self.ffn(x, **kwargs), **kwargs)
        return x
