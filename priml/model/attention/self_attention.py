"""Single-stream multi-head self-attention."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, override

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.distributed.tensor import DTensor

import torch

from priml.model.attention.kernel import SdpaFused
from priml.model.attention.kvcache import KVCache
from priml.model.attention.rope import RoPE
from priml.model.attention.window import causal_chunk_mask
from priml.model.custom_types import (
    AttentionKernel,
    ChannelsIn,
    DepthIndex,
    RotaryFactors,
)
from priml.model.init import InitFn, kaiming_uniform
from priml.model.linear import EnsembleLinear, Linear


class SelfAttention(nn.Module):
    """Multi-head self-attention with fused QKV EnsembleLinear.

    Uses a single EnsembleLinear for Q/K/V projections where each head
    is independently orthogonalizable by Muon. Supports GQA via num_heads_kv.
    """

    class Config(Fig["SelfAttention"], kw_only=False):
        """Set at least two of (channels_in, num_heads, channels_head); the third is inferred."""

        channels_in: int = -1
        """Model width (-1 to infer from num_heads * channels_head)."""

        channels_out: int = -1
        """Number of output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        num_heads: int = 8
        """Query-head count (-1 to infer from channels_in // channels_head)."""

        channels_head: int = -1
        """Per-head dimension (-1 to infer from channels_in // num_heads)."""

        num_heads_kv: int = -1
        """Key/value-head count for GQA (-1 = same as num_heads)."""

        bias: bool = False
        """Include bias in QKV and output projections."""

        dropout: float = 0.0
        """Attention dropout probability."""

        causal: bool = False
        """Apply causal (autoregressive) attention mask."""

        rope: Makeable[RotaryFactors] | None = None
        """Rotary position embedding (None = no positional encoding)."""

        norm_qk: Makeable[nn.Module] | None = None
        """Optional norm applied to Q and K before attention."""

        share_qk_norm: bool = True
        """Reuse one ``norm_qk`` instance for both Q and K.

        When False, two independent modules are built from the same
        config -- required for HF-format Qwen3 (separate ``q_norm`` /
        ``k_norm`` weights). When True, a single instance is shared
        (legacy behavior, half the params).
        """

        norm_out: Makeable[nn.Module] | None = None
        """Optional norm applied to attention output before proj_out."""

        split_qkv_projection: bool = False
        """Run Q/K/V projections as separate matmuls.

        Keep this disabled for normal use: the fused projection is the
        loop-native path. The split path exists for HuggingFace parity tests,
        where matching HF's operation order avoids small floating-point drift.
        """

        attn_kernel: Makeable[AttentionKernel] = field(
            default_factory=SdpaFused.Config,
        )
        """Attention kernel (SdpaFused or SdpaNaive)."""

        init_weight: InitFn = kaiming_uniform
        """Weight initialization function."""

        depth_index: DepthIndex = ()
        """Block depth index for depth-scaled init (-1 = no scaling)."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            self.channels_in, self.num_heads, self.channels_head = _infer_head_dims(
                channels_in=self.channels_in,
                num_heads=self.num_heads,
                channels_head=self.channels_head,
            )
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            if self.num_heads_kv == -1:
                self.num_heads_kv = self.num_heads
            if isinstance(self.norm_qk, ChannelsIn) and self.norm_qk.channels_in == -1:
                self.norm_qk.channels_in = self.channels_head
            if (
                isinstance(self.norm_out, ChannelsIn)
                and self.norm_out.channels_in == -1
            ):
                self.norm_out.channels_in = self.num_heads * self.channels_head
            return super().finalize()

    def __init__(self, config: Config) -> None:
        _validate_head_dims(config)
        if (
            -1 not in (config.channels_in, config.channels_out)
            and config.channels_in != config.channels_out
        ):
            raise ValueError(
                f"channels_in={config.channels_in} must equal "
                f"channels_out={config.channels_out} for SelfAttention."
            )
        super().__init__()
        if config.num_heads % config.num_heads_kv != 0:
            raise ValueError(
                f"num_heads={config.num_heads} must be divisible by "
                f"num_heads_kv={config.num_heads_kv}.",
            )
        self.num_heads = config.num_heads
        self.channels_head = config.channels_head
        self.num_heads_kv = config.num_heads_kv
        self.kv_groups = config.num_heads // config.num_heads_kv
        self.dropout = config.dropout
        self.causal = config.causal
        self.depth_index = config.depth_index
        self.split_qkv_projection = config.split_qkv_projection

        c = config.channels_in
        # Residual width (c) is decoupled from attention inner width
        # (num_heads * channels_head): Qwen3 sets an explicit head_dim where
        # hidden != num_heads * head_dim. proj_qkv reads the residual stream;
        # proj_out maps the concatenated num_heads back to it.
        inner = config.num_heads * config.channels_head
        # Fused QKV: one EnsembleLinear, each head orthogonalized independently by Muon.
        self.proj_qkv = EnsembleLinear.Config(
            channels_in=c,
            channels_out=config.channels_head,
            num_ensemble=config.num_heads + 2 * config.num_heads_kv,
            bias=config.bias,
            depth_index=config.depth_index,
            init_weight=config.init_weight,
            shard="colwise",
        ).make()
        self.proj_out = Linear.Config(
            channels_in=inner,
            channels_out=c,
            bias=config.bias,
            depth_index=config.depth_index,
            init_weight=config.init_weight,
            shard="rowwise",
        ).make()

        if config.norm_qk is None:
            self.norm_q: nn.Module | None = None
            self.norm_k: nn.Module | None = None
        elif config.share_qk_norm:
            shared = config.norm_qk.make()
            self.norm_q = shared
            self.norm_k = shared
        else:
            self.norm_q = config.norm_qk.make()
            self.norm_k = config.norm_qk.make()
        self.norm_out = config.norm_out.make() if config.norm_out else None
        self.rope = config.rope.make() if config.rope else None
        self.attn_kernel = config.attn_kernel.make()

    def assert_tensor_parallel_compatible(self) -> None:
        """Reject the fused flash kernel when this block is sharded.

        ``F.scaled_dot_product_attention`` dispatches to a flash kernel that
        has no DTensor sharding strategy, so it raises a cryptic deep-stack
        error on sharded q/k/v. ``SdpaNaive`` decomposes into matmul/softmax,
        which DTensor supports.
        """
        if isinstance(self.attn_kernel, SdpaFused) and isinstance(
            self.proj_qkv.weight,
            DTensor,
        ):
            raise RuntimeError(  # noqa: TRY004  -- unsupported config, not a type error
                "Tensor parallelism requires a DTensor-compatible attention "
                "kernel; set attn_kernel=SdpaNaive (the fused flash kernel has "
                "no DTensor sharding strategy).",
            )

    def reset_parameters(self) -> None:
        self.proj_qkv.reset_parameters()
        self.proj_out.reset_parameters()
        # Identity-check skips the shared-norm case (one reset, not two).
        seen: nn.Module | None = None
        for norm in (self.norm_q, self.norm_k):
            if norm is not None and norm is not seen:
                if hasattr(norm, "reset_parameters"):
                    norm.reset_parameters()
                seen = norm
        if self.norm_out and hasattr(self.norm_out, "reset_parameters"):
            self.norm_out.reset_parameters()

    def alloc_kv_cache(
        self,
        *,
        batch: int | tuple[int, ...],
        max_seq: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KVCache:
        """Allocate a KV cache sized for this attention block."""
        return KVCache.alloc(
            batch=batch,
            num_heads=self.num_heads_kv,
            max_seq=max_seq,
            channels_head=self.channels_head,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(
        self,
        x: Tensor,
        *,
        positions: Tensor | list[Tensor] | None = None,
        cos_sin: tuple[Tensor, Tensor] | None = None,
        dropout_p: float | None = None,
        is_causal: bool | None = None,
        attn_mask: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        out, _ = self._forward(
            x,
            positions=positions,
            cos_sin=cos_sin,
            cache=None,
            dropout_p=dropout_p,
            is_causal=is_causal,
            attn_mask=attn_mask,
            **kwargs,
        )
        return out

    def forward_cached(
        self,
        x: Tensor,
        *,
        cache: KVCache,
        positions: Tensor | list[Tensor] | None = None,
        cos_sin: tuple[Tensor, Tensor] | None = None,
        dropout_p: float | None = None,
        is_causal: bool | None = None,
        attn_mask: Tensor | None = None,
        **kwargs: object,
    ) -> tuple[Tensor, KVCache]:
        """Attend using and updating ``cache``."""
        out, updated = self._forward(
            x,
            positions=positions,
            cos_sin=cos_sin,
            cache=cache,
            dropout_p=dropout_p,
            is_causal=is_causal,
            attn_mask=attn_mask,
            **kwargs,
        )
        assert updated is not None
        return out, updated

    def _forward(
        self,
        x: Tensor,
        *,
        positions: Tensor | list[Tensor] | None,
        cos_sin: tuple[Tensor, Tensor] | None,
        cache: KVCache | None,
        dropout_p: float | None,
        is_causal: bool | None,
        attn_mask: Tensor | None,
        **kwargs: object,
    ) -> tuple[Tensor, KVCache | None]:
        S = x.shape[-2]

        # proj_qkv: [..., S, C] -> [..., S, num_ensemble, channels_head]
        if self.split_qkv_projection:
            q, k, v = self._split_qkv(x)
        else:
            q, k, v = (
                t.contiguous()
                for t in self.proj_qkv(x).split(
                    [self.num_heads, self.num_heads_kv, self.num_heads_kv],
                    dim=-2,
                )
            )

        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = RoPE.rotate(q, k, cos, sin)
        elif self.rope is not None:
            if positions is None:
                offset = cache.seen if cache is not None else 0
                positions = torch.arange(offset, offset + S, device=x.device)
            assert isinstance(positions, Tensor)
            cos, sin = self.rope(positions)
            q, k = RoPE.rotate(q, k, cos, sin)

        # The cache stores [..., H, S, D]; the kernels take [..., S, H, D].
        q, k, v = (t.movedim(-3, -2) for t in (q, k, v))
        if cache is not None:
            k, v = cache.update(k, v)

        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=-3)
            v = v.repeat_interleave(self.kv_groups, dim=-3)
        q, k, v = (t.movedim(-3, -2) for t in (q, k, v))

        out = self.attn_kernel(
            q,
            k,
            v,
            dropout_p=(
                (self.dropout if self.training else 0.0)
                if dropout_p is None
                else dropout_p
            ),
            is_causal=(
                self.causal and k.shape[-3] == S if is_causal is None else is_causal
            ),
            attn_mask=(
                causal_chunk_mask(q, k)
                if attn_mask is None and self.causal
                else attn_mask
            ),
            **kwargs,
        )

        # [..., S, H, D] -> [..., S, H*D]
        out = out.flatten(-2)
        if self.norm_out is not None:
            out = self.norm_out(out)
        out = self.proj_out(out)
        return out, cache

    def _split_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project Q/K/V separately while reusing the fused parameter layout."""
        c = self.channels_head
        q_end = self.num_heads
        k_end = q_end + self.num_heads_kv
        w = self.proj_qkv.weight.to(x.dtype)
        bias = self.proj_qkv.bias
        b = bias.to(x.dtype) if bias is not None else None
        q_w = w[:q_end].reshape(q_end * c, -1)
        k_w = w[q_end:k_end].reshape(self.num_heads_kv * c, -1)
        v_w = w[k_end:].reshape(self.num_heads_kv * c, -1)
        q = torch.matmul(x, q_w.T)
        k = torch.matmul(x, k_w.T)
        v = torch.matmul(x, v_w.T)
        if b is not None:
            q = q + b[:q_end].reshape(q_end * c)
            k = k + b[q_end:k_end].reshape(self.num_heads_kv * c)
            v = v + b[k_end:].reshape(self.num_heads_kv * c)
        q = q.reshape(*x.shape[:-1], self.num_heads, c)
        k = k.reshape(*x.shape[:-1], self.num_heads_kv, c)
        v = v.reshape(*x.shape[:-1], self.num_heads_kv, c)
        return q, k, v


def _infer_head_dims(
    *,
    channels_in: int,
    num_heads: int,
    channels_head: int,
) -> tuple[int, int, int]:
    """Infer one missing boundary or uniform-head dimension."""
    if channels_in == -1 and num_heads > 0 and channels_head > 0:
        channels_in = num_heads * channels_head
    if (
        channels_head == -1
        and channels_in != -1
        and num_heads > 0
        and channels_in % num_heads == 0
    ):
        channels_head = channels_in // num_heads
    if (
        num_heads == -1
        and channels_in != -1
        and channels_head > 0
        and channels_in % channels_head == 0
    ):
        num_heads = channels_in // channels_head
    return channels_in, num_heads, channels_head


def _validate_head_dims(config: SelfAttention.Config) -> None:
    """Reject geometry that stayed unresolved after finalize.

    Only the inference contract, never a dimension's sign: torch raises on a
    negative extent when it builds the tensor, and re-checking here would add a
    second message for one fault (STYLE.md "Let the leaf complain").
    """
    if config.channels_head == -1 and config.channels_in != -1:
        raise ValueError(
            f"channels_in={config.channels_in} not divisible by "
            f"num_heads={config.num_heads}; set channels_head explicitly.",
        )
    if config.num_heads == -1 and config.channels_in != -1:
        raise ValueError(
            f"channels_in={config.channels_in} not divisible by "
            f"channels_head={config.channels_head}; set num_heads explicitly.",
        )
    if -1 in (config.channels_in, config.num_heads, config.channels_head):
        raise ValueError(
            f"Need at least two of channels_in={config.channels_in}, "
            f"num_heads={config.num_heads}, channels_head={config.channels_head}.",
        )
