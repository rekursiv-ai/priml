"""Transformer block."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Any, Self, cast, overload, override

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

import torch

from priml.model.attention import SelfAttention
from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    HasDepth,
    HeadGeometry,
    ShardableConfig,
    TensorModule,
    propagate_attr,
)
from priml.model.kvcache import KVCache
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU


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

        depth: int = -1
        """Block depth index for depth-scaled init (-1 = no scaling)."""

        @property
        def channels_out(self) -> int:
            return self.channels_in

        @property
        def heads(self) -> int:
            """Attention heads, from the sublayer that owns them."""
            return self.attn.heads if isinstance(self.attn, HeadGeometry) else 1

        @property
        def channels_head(self) -> int:
            """Channels per head, from the sublayer that owns them."""
            if isinstance(self.attn, HeadGeometry):
                return self.attn.channels_head
            return self.channels_in

        @override
        def finalize(self) -> Self:
            c = self.channels_in
            for cfg in (self.attn, self.ffn, self.norm1, self.norm2):
                propagate_attr(cfg, "channels_in", c, protocol=ChannelsIn)
                propagate_attr(cfg, "channels_out", c, protocol=ChannelsOut)
                propagate_attr(cfg, "depth", self.depth, protocol=HasDepth)
            # Tensor parallelism: the FFN shards over the tp dim (its block-
            # internal style handles the fused-gate split alignment). The
            # attention block's children self-declare their styles.
            if isinstance(self.ffn, ShardableConfig):
                self.ffn.shard = "colwise"
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.prenorm = config.prenorm
        self.checkpoint = config.checkpoint
        self.depth = config.depth
        self.attn = config.attn.make()
        self.ffn = config.ffn.make()
        self.norm1 = config.norm1.make()
        self.norm2 = config.norm2.make()

    def reset_parameters(self) -> None:
        for m in (self.attn, self.ffn, self.norm1, self.norm2):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    @overload
    def __call__(
        self, x: Tensor, *args: Any, cache: KVCache
    ) -> tuple[Tensor, KVCache]: ...
    @overload
    def __call__(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor: ...
    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, KVCache]:
        return cast(  # pyright: ignore[reportUnnecessaryCast] -- ty cannot resolve the generic `Module.__call__` here; pyright can
            "Tensor | tuple[Tensor, KVCache]", super().__call__(*args, **kwargs)
        )

    @override
    def forward(
        self,
        x: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor | tuple[Tensor, KVCache]:
        # Gate on ``torch.is_grad_enabled()``: activation checkpointing only saves
        # memory by recomputing in backward, so it is pointless with grad off
        # (eval / ``torch.no_grad`` / ``torch.inference_mode``) -- and wrapping a
        # block in ``torch.utils.checkpoint`` under ``inference_mode`` can deadlock
        # a multi-rank eval. ``is_grad_enabled()`` is the precise condition (a
        # backward will run); it subsumes the older ``x.requires_grad`` check and
        # also covers a ``requires_grad`` input inside a ``no_grad`` region.
        if self.checkpoint and torch.is_grad_enabled():
            return torch_checkpoint(
                self._forward,
                x,
                *args,
                use_reentrant=False,
                **kwargs,
            )
        return self._forward(x, *args, **kwargs)

    @staticmethod
    def _unpack(
        result: Tensor | tuple[Tensor, KVCache],
    ) -> tuple[Tensor, KVCache | None]:
        """Unpack attention output (may be (tensor, cache) tuple)."""
        if isinstance(result, Tensor):
            return result, None
        x, cache = result
        return x, cache

    def _forward(
        self,
        x: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor | tuple[Tensor, KVCache]:
        cache: KVCache | None = None
        if self.prenorm:
            attn_out, cache = self._unpack(self.attn(self.norm1(x), *args, **kwargs))
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
        else:
            attn_out, cache = self._unpack(self.attn(x, *args, **kwargs))
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self.ffn(x))
        if "cache" in kwargs and cache is not None:
            return x, cache
        return x
