"""Gated grouped-query attention with native projections and explicit cache state."""

from __future__ import annotations

from dataclasses import KW_ONLY, field, replace
from typing import Self, override

import math

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    matmul_cost,
    resolve_dtype,
)
from priml.model.attention.kernel import SdpaNaive, attention_kernel_cost
from priml.model.attention.kvcache import KVCache
from priml.model.attention.rope import rotation_cost
from priml.model.attention.window import causal_chunk_mask, window_mask
from priml.model.custom_types import (
    AttentionKernel,
    ChannelsIn,
    DepthIndex,
    HasResetParameters,
    RotaryFactors,
    TensorModule,
    infer_same_width,
)
from priml.model.init import InitFn, kaiming_uniform
from priml.model.legacy_keys import absorb_legacy_keys
from priml.model.linear import Linear
from priml.model.norm import CenteredRMSNorm


class GatedSelfAttention(nn.Module):
    """Apply sigmoid output gating before the output projection."""

    class Config(Fig["GatedSelfAttention"]):
        channels_in: int = -1
        """Input channel width."""

        channels_out: int = -1
        """Output channel width."""

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
            infer_same_width(self)
            if isinstance(self.norm_qk, ChannelsIn) and self.norm_qk.channels_in == -1:
                self.norm_qk.channels_in = self.channels_head
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Price four projections, two norms, rotary, the kernel, and the gate.

            The query projection emits the gate beside the queries, so it is
            twice the query width. Each norm runs over its own head rows and
            owns one scale. The gate is a sigmoid and a product per inner
            channel; its adjoint pulls a gradient back through both factors.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            rows = seq_len * batch_size
            dt = dtype
            inner = self.num_heads * self.channels_head
            kv = self.num_heads_kv * self.channels_head
            total = (
                matmul_cost(
                    channels_in=self.channels_in,
                    channels_out=2 * inner,
                    bias=self.bias,
                    dtype=dt,
                    rows=rows,
                )
                + matmul_cost(
                    channels_in=self.channels_in,
                    channels_out=kv,
                    bias=self.bias,
                    dtype=dt,
                    rows=rows,
                ).tile(2, copies=2)
                + matmul_cost(
                    channels_in=inner,
                    channels_out=self.channels_out,
                    bias=self.bias,
                    dtype=dt,
                    rows=rows,
                )
            )
            for heads in (self.num_heads, self.num_heads_kv):
                total += cost(
                    self.norm_qk,
                    seq_len=seq_len * heads,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                ).tile(heads)
            if self.rope is not None:
                total += cost(
                    self.rope,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                total += rotation_cost(
                    self.rope,
                    rows=rows,
                    dtype=dt,
                    channels_head=self.channels_head,
                    heads=self.num_heads + self.num_heads_kv,
                )
            total += attention_kernel_cost(
                seq_len=seq_len,
                dtype=dtype,
                num_heads=self.num_heads,
                channels_head=self.channels_head,
                dropout_p=self.dropout,
            )
            total += elementwise_cost(
                primal=5 * inner,
                adjoint=6 * inner,
                channels=inner,
                inputs=4,
                outputs=1,
                adjoint_inputs=9,
                adjoint_outputs=3,
                rows=rows,
                dtype=dt,
            )
            return replace(total, bytes_state=resolve_dtype(dtype).itemsize * 2 * kv)

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
        proj_q = Linear.Config()
        proj_q.channels_in = config.channels_in
        proj_q.channels_out = 2 * config.num_heads * config.channels_head
        proj_q.bias = config.bias
        proj_q.init_weight = config.init_weight
        self.proj_q = proj_q.make()
        proj_k = Linear.Config()
        proj_k.channels_in = config.channels_in
        proj_k.channels_out = config.num_heads_kv * config.channels_head
        proj_k.bias = config.bias
        proj_k.init_weight = config.init_weight
        self.proj_k = proj_k.make()
        proj_v = Linear.Config()
        proj_v.channels_in = config.channels_in
        proj_v.channels_out = config.num_heads_kv * config.channels_head
        proj_v.bias = config.bias
        proj_v.init_weight = config.init_weight
        self.proj_v = proj_v.make()
        proj_out = Linear.Config()
        proj_out.channels_in = config.num_heads * config.channels_head
        proj_out.channels_out = config.channels_out
        proj_out.bias = config.bias
        proj_out.init_weight = config.init_weight
        proj_out.depth_index = config.depth_index
        self.proj_out = proj_out.make()
        self.norm_q = config.norm_qk.make()
        self.norm_k = config.norm_qk.make()
        self.rope = config.rope.make() if config.rope is not None else None
        absorb_legacy_keys(
            self,
            {
                "q_proj": "proj_q",
                "k_proj": "proj_k",
                "v_proj": "proj_v",
                "out_proj": "proj_out",
            },
        )
        self.attn_kernel = config.attn_kernel.make()

    def reset_parameters(self) -> None:
        """Reset projection and normalization parameters."""
        for module in (
            self.proj_q,
            self.proj_k,
            self.proj_v,
            self.proj_out,
            self.norm_q,
            self.norm_k,
        ):
            module.reset_parameters()
        if isinstance(self.rope, HasResetParameters):
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
            device=self.proj_q.weight.device if device is None else device,
            dtype=self.proj_q.weight.dtype if dtype is None else dtype,
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
            x,
            cache=cache,
            positions=positions,
            attn_mask=attn_mask,
            **kwargs,
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
        kwargs.pop("is_causal", None)
        shape = (*x.shape[:-1], -1, self.channels_head)
        q, gate = (
            self.proj_q(x)
            .reshape(
                *x.shape[:-1],
                self.num_heads,
                2 * self.channels_head,
            )
            .chunk(2, dim=-1)
        )
        q = self.norm_q(q)
        k = self.norm_k(self.proj_k(x).reshape(shape))
        v = self.proj_v(x).reshape(shape)
        if self.rope is not None:
            if positions is None:
                offset = cache.seen if cache is not None else 0
                positions = torch.arange(offset, offset + x.shape[-2], device=x.device)
            elif not _is_text_positions(positions, x=x):
                raise ValueError(
                    "positions must be a text layout of [sequence] or "
                    "[batch..., sequence, 1].",
                )
            cos, sin = self.rope(positions)
            q = _rotate(q, cos=cos.to(x.dtype), sin=sin.to(x.dtype))
            k = _rotate(k, cos=cos.to(x.dtype), sin=sin.to(x.dtype))
        if cache is not None:
            k, v = cache.update(k.movedim(-3, -2), v.movedim(-3, -2))
            k, v = k.movedim(-3, -2), v.movedim(-3, -2)
        groups = self.num_heads // self.num_heads_kv
        k, v = k.repeat_interleave(groups, dim=-2), v.repeat_interleave(groups, dim=-2)
        if attn_mask is None:
            is_causal = k.shape[-3] == q.shape[-3]
            # A window reaching the whole context returns None. In a rectangular
            # cached chunk, preserve the chunk's causal mask in that case; the
            # square fast path still keeps both masks absent.
            window = kwargs.get("window", -1)
            assert isinstance(window, int)
            mask = window_mask(q, k, window=window)
            if mask is None:
                mask = causal_chunk_mask(q, k)
        else:
            # A caller's mask (e.g. Qwen 3.5's padding mask) fills only WITHIN
            # the causal cone and leaves the rest at 0, trusting causality to
            # be enforced separately -- so it must still be combined with a
            # full causal mask here, not just passed through. It cannot lean
            # on `is_causal` for that: the kernels' fast path fills with a
            # literal -inf (this module's own convention -- see window_mask,
            # causal_chunk_mask, and SdpaNaive's own is_causal branch), but
            # Qwen 3.5's caller-supplied mask is `finfo.min`-filled to match
            # its HF reference bit-exactly. Mixing the two changes which value
            # wins a fully-masked row's softmax -- measured against the Qwen
            # 3.5 HF reference, a query whose only causally valid key is
            # itself masked-out collapses to 100% weight on that (masked) key
            # instead of HF's uniform fallback. That guarantee holds for a
            # 2-D padding mask (`_full_attention_mask` fills only within the
            # causal cone, so nothing double-fills); a caller-supplied 4-D
            # prepared mask already carries its own causal fill, so this
            # double-fills its non-causal cells -- confirmed confined to
            # padding-query rows, with zero leakage into live-row logits, but
            # the uniform-fallback guarantee above does not extend to it. So
            # this branch deliberately diverges from the module's usual -inf
            # and matches the caller's finfo.min instead.
            #
            # `window` must be read out of kwargs and applied here too: the
            # kernels only ever build their own window_mask when attn_mask is
            # None, so once a caller supplies one, window would otherwise be
            # silently ignored a second way.
            is_causal = False
            window = kwargs.get("window", -1)
            assert isinstance(window, int)
            mask = _causal_bias(q, k, dtype=x.dtype, window=window) + attn_mask
        output = (
            self.attn_kernel(
                q,
                k,
                v,
                is_causal=is_causal,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                **kwargs,
            )
            .flatten(-2)
            .contiguous()
        )
        return self.proj_out(output * gate.flatten(-2).sigmoid())


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
            "the input and attention configuration.",
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
            f"Rotary width {width} exceeds attention head width {x.shape[-1]}.",
        )
    rotated, passthrough = x[..., :width], x[..., width:]
    first, second = rotated.chunk(2, dim=-1)
    rotated = rotated * cos + torch.cat((-second, first), dim=-1) * sin
    return torch.cat((rotated, passthrough), dim=-1).movedim(-3, -2)


# Unlike causal_chunk_mask/window_mask, this always materializes and never
# uses a literal -inf: it exists only to add onto a caller-supplied mask,
# and that mask is finite-filled (see the `forward` comment on why the two
# fills cannot mix). Takes `window` too, since a caller-supplied mask stops
# the kernel from ever reaching its own window_mask.
def _causal_bias(
    q: Tensor,
    k: Tensor,
    *,
    dtype: torch.dtype,
    window: int = -1,
) -> Tensor:
    """Full additive causal mask, optionally windowed, finite-filled."""
    s, t = q.shape[-3], k.shape[-3]
    offset = torch.arange(t, device=q.device)
    offset = offset[t - s :, None] - offset[None, :]
    allowed = offset >= 0
    if window >= 0:
        allowed = allowed & (offset <= window)
    return torch.zeros(s, t, dtype=dtype, device=q.device).masked_fill(
        ~allowed,
        torch.finfo(dtype).min,
    )
