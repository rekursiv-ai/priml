"""Transformer block."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from functools import partial
from typing import Self, cast, override

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

import torch

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
)
from priml.model.attention.self_attention import SelfAttention
from priml.model.custom_types import (
    CachedAttention,
    ChannelsIn,
    ChannelsOut,
    DepthIndex,
    HasDepthIndex,
    HeadGeometry,
    Shardable,
    TensorModule,
    infer_same_width,
    propagate_attr,
)
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """Transformer block with attention and FFN.

    Module names match sic convention for checkpoint compatibility:
    ``attn``, ``ffn``, and their corresponding ``norm1`` and ``norm2``.
    """

    class Config(Fig["TransformerBlock"], kw_only=False):
        channels_in: int = -1
        """Input channel width."""

        channels_out: int = -1
        """Output channel width."""

        _: KW_ONLY

        attn: Makeable[TensorModule] = field(default_factory=SelfAttention.Config)
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
        """Block depth index for depth-scaled init (empty = no scaling)."""

        @property
        def num_heads(self) -> int:
            """Return the attention sublayer's head count."""
            return cast(HeadGeometry, self.attn).num_heads

        @property
        def channels_head(self) -> int:
            """Return the attention sublayer's per-head channel width."""
            return cast(HeadGeometry, self.attn).channels_head

        @override
        def finalize(self) -> Self:
            infer_same_width(self)
            c = self.channels_in
            for cfg in (self.attn, self.ffn, self.norm1, self.norm2):
                if isinstance(cfg, ChannelsIn) and cfg.channels_in == -1:
                    propagate_attr(cfg, "channels_in", c, protocol=ChannelsIn)
                if isinstance(cfg, ChannelsOut) and cfg.channels_out == -1:
                    propagate_attr(cfg, "channels_out", c, protocol=ChannelsOut)
                if (
                    self.depth_index
                    and isinstance(cfg, HasDepthIndex)
                    and not cfg.depth_index
                ):
                    propagate_attr(
                        cfg,
                        "depth_index",
                        self.depth_index,
                        protocol=HasDepthIndex,
                    )
            # Tensor parallelism: the FFN shards over the tp dim (its block-
            # internal style handles the fused-gate split alignment). The
            # attention block's children self-declare their styles.
            if isinstance(self.ffn, Shardable) and self.ffn.shard is None:
                self.ffn.shard = "colwise"
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Sum the four sublayers; ``checkpoint`` is recompute, not model work.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            residual_adds = elementwise_cost(
                primal=2 * self.channels_in,
                adjoint=2 * self.channels_in,
                channels=2 * self.channels_in,
                inputs=2,
                dtype=dtype,
            )
            return sum(
                (
                    cost(
                        child,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        dtype=dtype,
                        **kwargs,
                    )
                    for child in (self.attn, self.ffn, self.norm1, self.norm2)
                ),
                residual_adds,
            )

    def __init__(self, config: Config) -> None:
        # A one-channel FFN would silently broadcast across the residual stream.
        if (
            isinstance(config.ffn, ChannelsOut)
            and config.ffn.channels_out != config.channels_in
        ):
            raise ValueError(
                f"ffn.channels_out={config.ffn.channels_out} must equal "
                f"channels_in={config.channels_in} for TransformerBlock.",
            )
        if (
            isinstance(config.attn, ChannelsOut)
            and config.attn.channels_out != config.channels_in
        ):
            raise ValueError(
                f"attn.channels_out={config.attn.channels_out} must equal "
                f"channels_in={config.channels_in} for TransformerBlock.",
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
        """Initialize every parameter in place."""
        for m in (self.attn, self.ffn, self.norm1, self.norm2):
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

    def forward_cached[CacheT](
        self,
        x: Tensor,
        *,
        cache: CacheT,
        **kwargs: object,
    ) -> tuple[Tensor, CacheT]:
        """Run the block while updating its attention cache.

        Args:
          x: Input tensor.
          cache: Attention cache passed to the attention module.
          **kwargs: Additional arguments forwarded to attention and FFN.

        Returns:
          output: Output tensor same shape as x.
          cache: Updated cache after attention.

        """
        if not isinstance(self.attn, CachedAttention):
            raise TypeError("The attention module must implement cached attention.")
        attention = cast(CachedAttention[CacheT], self.attn)
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
            attn_out = self.attn(self.norm1(x, **kwargs), **kwargs)
            x = x + attn_out
            x = x + self.ffn(self.norm2(x, **kwargs), **kwargs)
        else:
            attn_out = self.attn(x, **kwargs)
            x = self.norm1(x + attn_out, **kwargs)
            x = self.norm2(x + self.ffn(x, **kwargs), **kwargs)
        return x
