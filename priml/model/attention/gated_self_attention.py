"""Gated grouped-query attention with native projections and explicit cache state."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, override

import math

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.kvcache import KVCache
from priml.model.custom_types import (
    AttentionKernel,
    ChannelsIn,
    DepthIndex,
    Resettable,
    RotaryFactors,
    TensorModule,
)
from priml.model.init import InitFn, kaiming_uniform
from priml.model.linear import Linear
from priml.model.norm import CenteredRMSNorm


class GatedSelfAttention(nn.Module):
    """Apply sigmoid output gating before the output projection."""

    class Config(Fig["GatedSelfAttention"]):
        channels_in: int = -1
        """Input feature width."""

        channels_out: int = -1
        """Output feature width; -1 inherits channels_in."""

        _: KW_ONLY

        num_heads: int = 16
        """Query and gate head count."""

        num_heads_kv: int = 4
        """Key and value head count."""

        channels_head: int = 256
        """Width of each attention head."""

        bias: bool = False
        """Include bias in the four projections."""

        dropout: float = 0.0
        """Attention dropout during training."""

        norm_qk: Makeable[TensorModule] = field(default_factory=CenteredRMSNorm.Config)
        """Independent per-head query and key normalization."""

        rope: Makeable[RotaryFactors] | None = None
        """Rotary factors; their width determines the rotated prefix."""

        attn_kernel: Makeable[AttentionKernel] = field(default_factory=SdpaNaive.Config)
        """Kernel applied before sigmoid gating and the output projection."""

        init_weight: InitFn = kaiming_uniform
        """Projection initializer."""

        depth_index: DepthIndex = ()
        """Depth information for the output projection initializer."""

        @override
        def finalize(self) -> Self:
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            if isinstance(self.norm_qk, ChannelsIn) and self.norm_qk.channels_in == -1:
                self.norm_qk.channels_in = self.channels_head
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        if min(config.num_heads, config.num_heads_kv, config.channels_head) <= 0:
            raise ValueError("Attention head counts and widths must be positive.")
        if config.num_heads % config.num_heads_kv:
            raise ValueError("Query heads must be divisible by key/value heads.")
        if (
            not math.isfinite(config.dropout)
            or config.dropout < 0
            or config.dropout > 1
        ):
            raise ValueError("dropout must be finite and in [0, 1].")
        self.num_heads = config.num_heads
        self.num_heads_kv = config.num_heads_kv
        self.channels_head = config.channels_head
        self.dropout = config.dropout
        q_proj = Linear.Config()
        q_proj.channels_in = config.channels_in
        q_proj.channels_out = 2 * config.num_heads * config.channels_head
        q_proj.bias = config.bias
        q_proj.init_weight = config.init_weight
        self.q_proj = q_proj.make()
        k_proj = Linear.Config()
        k_proj.channels_in = config.channels_in
        k_proj.channels_out = config.num_heads_kv * config.channels_head
        k_proj.bias = config.bias
        k_proj.init_weight = config.init_weight
        self.k_proj = k_proj.make()
        v_proj = Linear.Config()
        v_proj.channels_in = config.channels_in
        v_proj.channels_out = config.num_heads_kv * config.channels_head
        v_proj.bias = config.bias
        v_proj.init_weight = config.init_weight
        self.v_proj = v_proj.make()
        out_proj = Linear.Config()
        out_proj.channels_in = config.num_heads * config.channels_head
        out_proj.channels_out = config.channels_out
        out_proj.bias = config.bias
        out_proj.init_weight = config.init_weight
        out_proj.depth_index = config.depth_index
        self.out_proj = out_proj.make()
        self.norm_q = config.norm_qk.make()
        self.norm_k = config.norm_qk.make()
        self.rope = config.rope.make() if config.rope is not None else None
        self.attn_kernel = config.attn_kernel.make()

    def reset_parameters(self) -> None:
        """Reset projection and normalization parameters."""
        for module in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.out_proj,
            self.norm_q,
            self.norm_k,
        ):
            module.reset_parameters()
        if isinstance(self.rope, Resettable):
            self.rope.reset_parameters()

    def alloc_kv_cache(
        self,
        *,
        batch: int | tuple[int, ...],
        max_seq: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KVCache:
        """Allocate an empty cache on the projection's device and dtype.

        Args:
          batch: Batch size or shape tuple.
          max_seq: Maximum number of cached tokens.
          device: Cache placement; defaults to the query projection's device.
          dtype: Cache dtype; defaults to the query projection's dtype.

        Returns:
          cache: Empty key/value cache sized for this attention module.

        """
        return KVCache.alloc(
            batch=batch,
            num_heads=self.num_heads_kv,
            max_seq=max_seq,
            channels_head=self.channels_head,
            device=self.q_proj.weight.device if device is None else device,
            dtype=self.q_proj.weight.dtype if dtype is None else dtype,
        )

    def forward_cached(
        self,
        x: Tensor,
        *,
        cache: KVCache,
        positions: Tensor | None = None,
        attn_mask: Tensor | None = None,
        **kwargs: object,
    ) -> tuple[Tensor, KVCache]:
        """Append tokens to cache and return their attention outputs.

        Args:
          x: Hidden states shaped [batch, sequence, channels].
          cache: Preallocated cache updated in place.
          positions: Optional supported text positions for rotary factors.
          attn_mask: Optional additive mask broadcastable to [batch, heads, Q, K].
          **kwargs: Messages for the injected attention kernel.

        Returns:
          output: Gated attention outputs for the new input tokens.
          cache: The updated input cache.

        """
        return self.forward(
            x, cache=cache, positions=positions, attn_mask=attn_mask, **kwargs
        ), cache

    @override
    def forward(
        self,
        x: Tensor,
        *,
        cache: KVCache | None = None,
        positions: Tensor | None = None,
        attn_mask: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Attend causally, gating each value channel before projecting out.

        Args:
          x: Hidden states shaped [batch, sequence, channels].
          cache: Optional preallocated cache, updated in place without eviction.
          positions: Text positions as ``[sequence]`` or
            ``[batch..., sequence, 1]``. The axis-last form must match x's
            batch and sequence axes; multiple position axes are unsupported.
          attn_mask: Optional additive mask broadcastable to [batch, heads, Q, K].
          **kwargs: Messages for the injected attention kernel.

        Returns:
          output: Gated attention outputs with the input's batch and sequence axes.

        """
        if cache is not None:
            _validate_cache_geometry(
                cache,
                x=x,
                num_heads_kv=self.num_heads_kv,
                channels_head=self.channels_head,
            )
            if cache.seen + x.shape[-2] > cache.max_seq:
                raise ValueError("The full-attention cache capacity would be exceeded.")
        shape = (*x.shape[:-1], -1, self.channels_head)
        q, gate = (
            self.q_proj(x)
            .reshape(
                *x.shape[:-1],
                self.num_heads,
                2 * self.channels_head,
            )
            .chunk(2, dim=-1)
        )
        q = self.norm_q(q)
        k = self.norm_k(self.k_proj(x).reshape(shape))
        v = self.v_proj(x).reshape(shape)
        if self.rope is not None:
            if positions is None:
                offset = cache.seen if cache is not None else 0
                positions = torch.arange(offset, offset + x.shape[-2], device=x.device)
            elif not _is_text_positions(positions, x=x):
                raise ValueError(
                    "positions must be a text layout of [sequence] or "
                    "[batch..., sequence, 1]."
                )
            cos, sin = self.rope(positions)
            q = _rotate(q, cos=cos.to(x.dtype), sin=sin.to(x.dtype))
            k = _rotate(k, cos=cos.to(x.dtype), sin=sin.to(x.dtype))
        if cache is not None:
            k, v = cache.update(k.movedim(-3, -2), v.movedim(-3, -2))
            k, v = k.movedim(-3, -2), v.movedim(-3, -2)
        groups = self.num_heads // self.num_heads_kv
        k, v = k.repeat_interleave(groups, dim=-2), v.repeat_interleave(groups, dim=-2)
        causal = torch.arange(k.shape[-3], device=x.device).unsqueeze(0) <= (
            torch.arange(q.shape[-3], device=x.device).unsqueeze(1)
            + k.shape[-3]
            - q.shape[-3]
        )
        mask = torch.zeros(causal.shape, dtype=x.dtype, device=x.device)
        mask = mask.masked_fill(~causal, torch.finfo(x.dtype).min)
        if attn_mask is not None:
            mask = mask + attn_mask
        output = (
            self.attn_kernel(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                **kwargs,
            )
            .flatten(-2)
            .contiguous()
        )
        return self.out_proj(output * gate.flatten(-2).sigmoid())


def _validate_cache_geometry(
    cache: KVCache,
    *,
    x: Tensor,
    num_heads_kv: int,
    channels_head: int,
) -> None:
    """Reject cache layouts that would broadcast or fail after mutation."""
    if (
        cache.k.ndim != x.ndim + 1
        or cache.v.ndim != x.ndim + 1
        or cache.k.shape[:-3] != x.shape[:-2]
        or cache.v.shape[:-3] != x.shape[:-2]
        or cache.k.shape[-3] != num_heads_kv
        or cache.v.shape[-3] != num_heads_kv
        or cache.k.shape[-2] != cache.v.shape[-2]
        or cache.k.shape[-1] != channels_head
        or cache.v.shape[-1] != channels_head
    ):
        raise ValueError(
            "Full-attention cache batch, head, and feature geometry must match "
            "the input and attention configuration."
        )


def _is_text_positions(positions: Tensor, *, x: Tensor) -> bool:
    """Return whether positions use one supported single-axis text layout."""
    if positions.ndim == 1:
        return positions.shape[0] == x.shape[-2]
    return positions.shape[-1] == 1 and positions.shape[:-1] == x.shape[:-1]


# Head-major layout preserves the upstream q/k norm weight-gradient reduction order. The
# generic RoPE rotation intentionally accumulates in float32; this gated-attention
# architecture rounds each product in activation dtype.
def _rotate(x: Tensor, *, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate in activation dtype and head-major layout, including backward."""
    x = x.movedim(-3, -2)
    cos, sin = cos.movedim(-3, -2), sin.movedim(-3, -2)
    cos, sin = torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)
    width = cos.shape[-1]
    if width > x.shape[-1]:
        raise ValueError(
            f"Rotary width {width} exceeds attention head width {x.shape[-1]}."
        )
    rotated, passthrough = x[..., :width], x[..., width:]
    first, second = rotated.chunk(2, dim=-1)
    rotated = rotated * cos + torch.cat((-second, first), dim=-1) * sin
    return torch.cat((rotated, passthrough), dim=-1).movedim(-3, -2)
