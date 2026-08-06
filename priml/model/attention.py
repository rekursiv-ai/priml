"""Multi-head attention: single-stream (SelfAttention) and N-stream (MultiStreamAttention)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, field
from typing import Any, Self, cast, override

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.distributed.tensor import DTensor
from torch.nn import functional as f

import torch

from priml.model.custom_types import ChannelsIn, ChannelsOut, propagate_attr
from priml.model.init import InitFn, kaiming_uniform
from priml.model.kvcache import KVCache
from priml.model.linear import EnsembleLinear, Linear
from priml.model.rope import RoPE


class SdpaFused(nn.Module):
    """Wraps F.scaled_dot_product_attention."""

    class Config(Fig["SdpaFused"]):
        pass

    def __init__(self, config: Config | None = None) -> None:
        del config
        super().__init__()

    @override
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        attn_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        del kwargs
        return f.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )


class SdpaNaive(nn.Module):
    """Manual matmul+softmax attention (matches HF eager_attention_forward)."""

    class Config(Fig["SdpaNaive"]):
        pass

    def __init__(self, config: Config | None = None) -> None:
        del config
        super().__init__()

    @override
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        attn_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        del kwargs
        scale = q.shape[-1] ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if is_causal:
            S, kS = q.shape[-2], k.shape[-2]
            mask = torch.ones(S, kS, dtype=torch.bool, device=q.device).tril(
                diagonal=kS - S,
            )
            attn = attn.masked_fill(~mask, float("-inf"))
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
        if dropout_p > 0.0:
            attn = f.dropout(attn, p=dropout_p)
        return torch.matmul(attn, v)


class SelfAttention(nn.Module):
    """Multi-head self-attention with fused QKV EnsembleLinear.

    Uses a single EnsembleLinear for Q/K/V projections where each head
    is independently orthogonalizable by Muon. Supports GQA via num_heads_kv.
    """

    class Config(Fig["SelfAttention"], kw_only=False):
        """Set at least two of (channels_in, heads, channels_head); the third is inferred."""

        channels_in: int = -1
        """Model width (-1 to infer from heads * channels_head)."""

        _: KW_ONLY

        heads: int = 8
        """Number of query attention heads (-1 to infer from channels_in // channels_head)."""

        channels_head: int = -1
        """Per-head dimension (-1 to infer from channels_in // heads)."""

        num_heads_kv: int = -1
        """Number of key/value heads for GQA (-1 = same as heads)."""

        bias: bool = False
        """Include bias in QKV and output projections."""

        dropout: float = 0.0
        """Attention dropout probability."""

        causal: bool = False
        """Apply causal (autoregressive) attention mask."""

        rope: RoPE.Config | None = None
        """Rotary position embedding config (None = no positional encoding)."""

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

        attn_kernel: Makeable[nn.Module] = field(
            default_factory=SdpaFused.Config,
        )
        """Attention kernel (SdpaFused or SdpaNaive)."""

        init_weight: InitFn = kaiming_uniform
        """Weight initialization function."""

        depth: int = -1
        """Block depth index for depth-scaled init (-1 = no scaling)."""

        @property
        def channels_out(self) -> int:
            return self.channels_in

        @override
        def finalize(self) -> Self:
            _infer_head_dims(self)
            if self.num_heads_kv == -1:
                self.num_heads_kv = self.heads
            if self.heads % self.num_heads_kv != 0:
                raise ValueError(
                    f"heads={self.heads} must be divisible by "
                    f"num_heads_kv={self.num_heads_kv}.",
                )
            if isinstance(self.norm_qk, ChannelsIn) and self.norm_qk.channels_in == -1:
                self.norm_qk.channels_in = self.channels_head
            if (
                isinstance(self.norm_out, ChannelsIn)
                and self.norm_out.channels_in == -1
            ):
                self.norm_out.channels_in = self.heads * self.channels_head
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.heads = config.heads
        self.channels_head = config.channels_head
        self.num_heads_kv = config.num_heads_kv
        self.kv_groups = config.heads // config.num_heads_kv
        self.dropout = config.dropout
        self.causal = config.causal
        self.depth = config.depth
        self.split_qkv_projection = config.split_qkv_projection

        c = config.channels_in
        # Residual width (c) is decoupled from attention inner width
        # (heads * channels_head): Qwen3 sets an explicit head_dim where
        # hidden != heads * head_dim. proj_qkv reads the residual stream;
        # proj_out maps the concatenated heads back to it.
        inner = config.heads * config.channels_head
        # Fused QKV: one EnsembleLinear, each head orthogonalized independently by Muon.
        self.proj_qkv = EnsembleLinear.Config(
            channels_in=c,
            channels_out=config.channels_head,
            num_ensemble=config.heads + 2 * config.num_heads_kv,
            bias=config.bias,
            depth=config.depth,
            init_weight=config.init_weight,
            shard="colwise",
        ).make()
        self.proj_out = Linear.Config(
            channels_in=inner,
            channels_out=c,
            bias=config.bias,
            depth=config.depth,
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
            heads=self.num_heads_kv,
            max_seq=max_seq,
            channels_head=self.channels_head,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(
        self,
        x: Tensor,
        *args: Any,
        positions: Tensor | list[Tensor] | None = None,
        cos_sin: tuple[Tensor, Tensor] | None = None,
        cache: KVCache | None = None,
        **kwargs: Any,
    ) -> Tensor | tuple[Tensor, KVCache]:
        del args, kwargs
        S = x.shape[-2]

        # proj_qkv: [..., S, C] -> [..., S, num_ensemble, channels_head]
        if self.split_qkv_projection:
            q, k, v = self._split_qkv(x)
        else:
            q, k, v = (
                t.contiguous()
                for t in self.proj_qkv(x).split(
                    [self.heads, self.num_heads_kv, self.num_heads_kv],
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
            cos, sin = self.rope(positions)
            q, k = RoPE.rotate(q, k, cos, sin)

        # [..., S, H, D] -> [..., H, S, D] for SDPA
        q, k, v = (t.movedim(-3, -2) for t in (q, k, v))

        if cache is not None:
            k, v = cache.update(k, v)
        else:
            cache = KVCache(k, v)

        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=-3)
            v = v.repeat_interleave(self.kv_groups, dim=-3)

        out = self.attn_kernel(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal and k.shape[-2] == S,
            attn_mask=_causal_chunk_mask(q, k) if self.causal else None,
        )

        # [..., H, S, D] -> [..., S, H*D]
        out = out.movedim(-3, -2).flatten(-2)
        if self.norm_out is not None:
            out = self.norm_out(out)
        out = self.proj_out(out)
        return out, cache

    def _split_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project Q/K/V separately while reusing the fused parameter layout."""
        c = self.channels_head
        q_end = self.heads
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
        q = q.reshape(*x.shape[:-1], self.heads, c)
        k = k.reshape(*x.shape[:-1], self.num_heads_kv, c)
        v = v.reshape(*x.shape[:-1], self.num_heads_kv, c)
        return q, k, v


def _infer_head_dims(config: Any) -> None:
    """Infer the head-dim triple ``(channels_in, heads, channels_head)``.

    Fills any single ``-1`` from the other two. ``channels_in`` is the
    residual width and ``heads * channels_head`` is the attention inner
    width; they may differ (Qwen3 sets an explicit ``head_dim``), so no
    equality is enforced. Inference of ``channels_head`` or ``heads`` by
    division must be exact.
    """
    if config.channels_in == -1 and config.heads != -1 and config.channels_head != -1:
        config.channels_in = config.heads * config.channels_head
    if config.channels_head == -1 and config.channels_in != -1 and config.heads != -1:
        if config.channels_in % config.heads != 0:
            raise ValueError(
                f"channels_in={config.channels_in} not divisible by "
                f"heads={config.heads}; set channels_head explicitly.",
            )
        config.channels_head = config.channels_in // config.heads
    if config.heads == -1 and config.channels_in != -1 and config.channels_head != -1:
        if config.channels_in % config.channels_head != 0:
            raise ValueError(
                f"channels_in={config.channels_in} not divisible by "
                f"channels_head={config.channels_head}; set heads explicitly.",
            )
        config.heads = config.channels_in // config.channels_head
    if -1 in (config.channels_in, config.heads, config.channels_head):
        raise ValueError(
            f"Need at least two of channels_in={config.channels_in}, "
            f"heads={config.heads}, channels_head={config.channels_head}.",
        )


def _causal_chunk_mask(q: Tensor, k: Tensor) -> Tensor | None:
    """Additive causal mask for a multi-token chunk against a longer cache.

    Returns ``None`` when query and key lengths match (the square case
    that ``is_causal`` already handles). Otherwise builds a ``[S, T]``
    bottom-right-aligned mask so query ``i`` (absolute position
    ``T - S + i``) attends to keys ``0..T - S + i`` -- never to later
    tokens in the same chunk.
    """
    s, t = q.shape[-2], k.shape[-2]
    if s == t:
        return None
    allowed = torch.ones(s, t, dtype=torch.bool, device=q.device).tril(diagonal=t - s)
    return torch.zeros(s, t, dtype=q.dtype, device=q.device).masked_fill(
        ~allowed,
        float("-inf"),
    )


class MultiStreamAttention(nn.Module):
    """N-stream joint attention with per-stream QKV and shared K/V.

    List-form of SelfAttention: each config field and forward kwarg
    mirrors SA but accepts one value per stream. K/V from all streams
    are concatenated so each stream's Q attends to the full set.
    """

    class Config(Fig["MultiStreamAttention"], kw_only=False):
        channels_in: int = -1
        """Channel width shared across all streams."""

        _: KW_ONLY

        num_streams: int = 2
        """Number of parallel token streams."""

        heads: int = 8
        """Number of query attention heads."""

        channels_head: int = -1
        """Per-head dimension (-1 = channels_in // heads)."""

        num_heads_kv: int = -1
        """Key/value heads for GQA (-1 = same as heads)."""

        bias: bool = False
        """Bias in QKV and output projections."""

        dropout: float = 0.0
        """Attention dropout probability."""

        causal: bool = False
        """Apply causal (autoregressive) attention mask."""

        rope: list[RoPE.Config | None] = field(
            default_factory=list[RoPE.Config | None],
        )
        """Per-stream RoPE configs (empty = no internal RoPE)."""

        norm_qk: Makeable[nn.Module] | None = None
        """Optional norm applied to Q and K before attention."""

        share_qk_norm: bool = True
        """Reuse one ``norm_qk`` instance for both Q and K.

        When False, two independent modules are built from the same
        config (shared across streams, but Q/K-independent).
        """

        norm_out: Makeable[nn.Module] | None = None
        """Optional norm applied to attention output before proj_out."""

        attn_kernel: Makeable[nn.Module] = field(
            default_factory=SdpaFused.Config,
        )
        """Attention kernel (SdpaFused or SdpaNaive)."""

        init_weight: InitFn = kaiming_uniform
        """Weight initialization function."""

        depth: int = -1
        """Block depth index for depth-scaled init."""

        @property
        def channels_out(self) -> int:
            return self.channels_in

        @override
        def finalize(self) -> Self:
            _infer_head_dims(self)
            if self.num_heads_kv == -1:
                self.num_heads_kv = self.heads
            if self.heads % self.num_heads_kv != 0:
                raise ValueError(
                    f"heads={self.heads} must be divisible by "
                    f"num_heads_kv={self.num_heads_kv}.",
                )
            if self.causal and self.num_streams > 1:
                raise ValueError(
                    "causal=True requires num_streams=1.",
                )
            if isinstance(self.norm_qk, ChannelsIn) and self.norm_qk.channels_in == -1:
                self.norm_qk.channels_in = self.channels_head
            if (
                isinstance(self.norm_out, ChannelsIn)
                and self.norm_out.channels_in == -1
            ):
                self.norm_out.channels_in = self.heads * self.channels_head
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_streams = config.num_streams
        self.heads = config.heads
        self.channels_head = config.channels_head
        self.num_heads_kv = config.num_heads_kv
        self.kv_groups = config.heads // config.num_heads_kv
        self.dropout = config.dropout
        self.causal = config.causal
        self.depth = config.depth

        c = config.channels_in
        N = config.num_streams
        # Residual width (c) decoupled from attention inner width.
        inner = config.heads * config.channels_head

        self.proj_qkvs = nn.ModuleList(
            EnsembleLinear.Config(
                channels_in=c,
                channels_out=config.channels_head,
                num_ensemble=config.heads + 2 * config.num_heads_kv,
                bias=config.bias,
                depth=config.depth,
                init_weight=config.init_weight,
                shard="colwise",
            ).make()
            for _ in range(N)
        )

        self.proj_outs = nn.ModuleList(
            Linear.Config(
                channels_in=inner,
                channels_out=c,
                bias=config.bias,
                depth=config.depth,
                init_weight=config.init_weight,
                shard="rowwise",
            ).make()
            for _ in range(N)
        )

        # Per-stream RoPE keyed by stream index; absent = no RoPE.
        self.ropes = nn.ModuleDict(
            {str(i): cfg.make() for i, cfg in enumerate(config.rope) if cfg is not None}
        )

        # QK norm: shared instance (``share_qk_norm=True``) or two
        # independent instances built from the same config. Shared
        # across streams in both modes (matches single-stream parity).
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
        self.norm_out: nn.Module | None = (
            config.norm_out.make() if config.norm_out else None
        )
        self.attn_kernel = config.attn_kernel.make()

    def assert_tensor_parallel_compatible(self) -> None:
        """Reject the fused flash kernel when this block is sharded.

        See :meth:`SelfAttention.assert_tensor_parallel_compatible`.
        """
        sharded = any(isinstance(qkv.weight, DTensor) for qkv in self.proj_qkvs)
        if isinstance(self.attn_kernel, SdpaFused) and sharded:
            raise RuntimeError(
                "Tensor parallelism requires a DTensor-compatible attention "
                "kernel; set attn_kernel=SdpaNaive (the fused flash kernel has "
                "no DTensor sharding strategy).",
            )

    def reset_parameters(self) -> None:
        for modules in (self.proj_qkvs, self.proj_outs):
            for m in modules:
                m.reset_parameters()
        for r in self.ropes.values():
            if hasattr(r, "reset_parameters"):
                r.reset_parameters()
        seen: nn.Module | None = None
        for norm in (self.norm_q, self.norm_k):
            if norm is not None and norm is not seen:
                if hasattr(norm, "reset_parameters"):
                    norm.reset_parameters()
                seen = norm
        if self.norm_out is not None and hasattr(self.norm_out, "reset_parameters"):
            self.norm_out.reset_parameters()

    @override
    def forward(
        self,
        xs: Sequence[Tensor],
        *args: Any,
        positions: Sequence[Tensor | list[Tensor] | None] | None = None,
        cos_sin: Sequence[tuple[Tensor, Tensor] | None] | None = None,
        cache: Sequence[KVCache | None] | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, ...] | tuple[tuple[Tensor, ...], list[KVCache]]:
        """Joint attention across N streams.

        Args:
          xs: Per-stream tokens [..., S_i, C].
          *args: Ignored (interface compat).
          positions: Per-stream position tensors for internal RoPE.
          cos_sin: Per-stream pre-computed RoPE (cos, sin) pairs.
            Overrides internal RoPE for that stream.
          cache: Per-stream KV caches.
          **kwargs: Ignored (interface compat).

        Returns:
          ys: Per-stream outputs (no cache), or (ys, caches) tuple.

        """
        del args, kwargs
        N = self.num_streams

        pos_list: list[Tensor | list[Tensor] | None] = (
            list(positions) if positions is not None else [None] * N
        )
        cs_list: list[tuple[Tensor, Tensor] | None] = (
            list(cos_sin) if cos_sin is not None else [None] * N
        )
        cache_list: list[KVCache | None] = (
            list(cache) if cache is not None else [None] * N
        )

        all_q: list[Tensor] = []
        all_k: list[Tensor] = []
        all_v: list[Tensor] = []
        out_caches: list[KVCache | None] = [None] * N

        for i, x in enumerate(xs):
            S = x.shape[-2]
            q, k, v = (
                t.contiguous()
                for t in self.proj_qkvs[i](x).split(
                    [self.heads, self.num_heads_kv, self.num_heads_kv],
                    dim=-2,
                )
            )

            norm_q = self.norm_q
            if norm_q is not None:
                q = norm_q(q)
            norm_k = self.norm_k
            if norm_k is not None:
                k = norm_k(k)

            # RoPE: prefer external cos_sin, fall back to internal rope.
            cs_i = cs_list[i]
            si = str(i)
            if cs_i is not None:
                cos, sin = cs_i
                q, k = RoPE.rotate(q, k, cos, sin)
            elif si in self.ropes:
                p = pos_list[i]
                if p is None:
                    c_i = cache_list[i]
                    offset = c_i.seen if c_i is not None else 0
                    p = torch.arange(offset, offset + S, device=x.device)
                cos, sin = self.ropes[si](p)
                q, k = RoPE.rotate(q, k, cos, sin)

            # [..., S, H, hd] → [..., H, S, hd] for SDPA.
            q, k, v = (t.movedim(-3, -2) for t in (q, k, v))

            c_i = cache_list[i]
            if c_i is not None:
                k, v = c_i.update(k, v)
            else:
                c_i = KVCache(k, v)
            out_caches[i] = c_i

            all_q.append(q)
            all_k.append(k)
            all_v.append(v)

        # Joint attention: concat K/V, each Q attends globally.
        k_cat = torch.cat(all_k, dim=-2)
        v_cat = torch.cat(all_v, dim=-2)

        if self.kv_groups > 1:
            k_cat = k_cat.repeat_interleave(self.kv_groups, dim=-3)
            v_cat = v_cat.repeat_interleave(self.kv_groups, dim=-3)

        total_kv = k_cat.shape[-2]
        results: list[Tensor] = []
        for i in range(N):
            S_i = all_q[i].shape[-2]
            out = self.attn_kernel(
                all_q[i],
                k_cat,
                v_cat,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal and total_kv == S_i,
                attn_mask=(
                    _causal_chunk_mask(all_q[i], k_cat) if self.causal else None
                ),
            )
            # [..., H, S, hd] → [..., S, H*hd].
            out = out.movedim(-3, -2).flatten(-2)
            if self.norm_out is not None:
                out = self.norm_out(out)
            out = self.proj_outs[i](out)
            results.append(out)

        outputs = tuple(results)
        if cache is not None:
            return outputs, [c for c in out_caches if c is not None]
        return outputs


class OutputGate(nn.Module):
    """Wrap an attention module with output gating.

    Computes ``gate = gate_proj(x)`` before delegating to ``inner``,
    then applies ``out * sigmoid(gate)`` to the attention output.
    Compatible with modules returning ``Tensor`` or
    ``tuple[Tensor, KVCache]``.
    """

    class Config(Fig["OutputGate"], kw_only=False):
        channels_in: int = -1
        """Model width for the gate projection."""

        _: KW_ONLY

        inner: Makeable[nn.Module] = field(default_factory=SelfAttention.Config)
        """Wrapped attention module config."""

        bias: bool = False
        """Include bias in the gate projection."""

        depth: int = -1
        """Block depth index for depth-scaled init (-1 = no scaling)."""

        @property
        def channels_out(self) -> int:
            return self.channels_in

        @override
        def finalize(self) -> Self:
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
            propagate_attr(self.inner, "depth", self.depth)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.inner = config.inner.make()
        self.gate_proj = Linear.Config(
            channels_in=config.channels_in,
            channels_out=config.channels_in,
            bias=config.bias,
        ).make()

    def reset_parameters(self) -> None:
        if hasattr(self.inner, "reset_parameters"):
            self.inner.reset_parameters()
        self.gate_proj.reset_parameters()

    @override
    def forward(
        self,
        x: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor | tuple[Tensor, KVCache]:
        gate = torch.sigmoid(self.gate_proj(x))
        result = self.inner(x, *args, **kwargs)
        if isinstance(result, tuple):
            out, cache = cast(tuple[Tensor, KVCache], result)
            return out * gate, cache
        return cast(Tensor, result) * gate
