"""Single-stream multi-head attention: self-attention, or cross-attention to a memory."""

from __future__ import annotations

from dataclasses import KW_ONLY, field, replace
from typing import Self, override

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn
from torch.distributed.tensor import DTensor

import torch

from priml.cost import (
    Cost,
    cost,
    matmul_cost,
    resolve_dtype,
)
from priml.model.attention.kernel import SdpaFused
from priml.model.attention.kvcache import KVCache
from priml.model.attention.rope import RoPE, rotation_cost
from priml.model.attention.window import causal_chunk_mask, window_mask
from priml.model.custom_types import (
    AttentionKernel,
    ChannelsIn,
    DepthIndex,
    HasResetParameters,
    LayerCache,
    RotaryConfig,
    TensorModule,
    infer_same_width,
)
from priml.model.init import InitFn, kaiming_uniform
from priml.model.linear import EnsembleLinear, Linear


class AttentionProjections(nn.Module):
    """Own attention projections, normalization and rotary encoding without a kernel.

    The configuration also declares attention policies, consumed by the owner
    applying these projections to either self-attention or joint attention.
    """

    class Config(Fig["AttentionProjections"], kw_only=False):
        """Set at least two of channels_in, num_heads and channels_head.

        The third is inferred.
        """

        channels_in: int = -1
        """Input channel width."""

        channels_out: int = -1
        """Output channel width."""

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

        rope: RotaryConfig | None = None
        """Rotary position embedding (None = no positional encoding)."""

        norm_qk: Makeable[TensorModule] | None = None
        """Optional norm applied to Q and K before attention."""

        share_qk_norm: bool = True
        """Reuse one ``norm_qk`` instance for both Q and K.

        When False, two independent modules are built from the same
        config -- required for HF-format Qwen3 (separate ``q_norm`` /
        ``k_norm`` weights). When True, a single instance is shared
        (legacy behavior, half the params).
        """

        norm_out: Makeable[TensorModule] | None = None
        """Optional norm applied to attention output before proj_out."""

        split_qkv_projection: bool = False
        """Run Q/K/V projections as separate matmuls.

        Keep this disabled for normal use: the fused projection is the
        loop-native path. The split path exists for HuggingFace parity tests,
        where matching HF's operation order avoids small floating-point drift.
        """

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
            if self.num_heads_kv == -1:
                self.num_heads_kv = self.num_heads
            if isinstance(self.norm_qk, ChannelsIn) and self.norm_qk.channels_in == -1:
                self.norm_qk.channels_in = self.channels_head
            if (
                isinstance(self.norm_out, ChannelsIn)
                and self.norm_out.channels_in == -1
            ):
                self.norm_out.channels_in = self.num_heads * self.channels_head
            infer_same_width(self)
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            memory_len: int = -1,
            **kwargs: object,
        ) -> Cost:
            """Cost the projections, norms, and rotary; no kernel here.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              memory_len: Memory positions per sequence the keys and values
                are projected from; -1 projects them from the sequence itself.
                Projections from a memory are not rotated.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation cost over ``seq_len`` and ``batch_size``.

            """
            rows = seq_len * batch_size
            dt = dtype
            inner = self.num_heads * self.channels_head
            if memory_len < 0:
                projection_heads = (
                    (self.num_heads, self.num_heads_kv, self.num_heads_kv)
                    if self.split_qkv_projection
                    else (self.num_heads + 2 * self.num_heads_kv,)
                )
                projections = [(heads, rows) for heads in projection_heads]
            else:
                projections = [
                    (self.num_heads, rows),
                    (2 * self.num_heads_kv, memory_len * batch_size),
                ]
            qkv = sum(
                (
                    matmul_cost(
                        channels_in=self.channels_in,
                        channels_out=heads * self.channels_head,
                        bias=self.bias,
                        dtype=dt,
                        rows=projected,
                    )
                    for heads, projected in projections
                ),
                Cost(),
            )
            out = matmul_cost(
                channels_in=inner,
                channels_out=self.channels_in,
                bias=self.bias,
                dtype=dt,
                rows=rows,
            )
            total = qkv + out
            if self.norm_qk is not None and memory_len >= 0:
                queries = cost(
                    self.norm_qk,
                    seq_len=seq_len,
                    batch_size=batch_size * self.num_heads,
                    dtype=dtype,
                    **kwargs,
                )
                keys = cost(
                    self.norm_qk,
                    seq_len=memory_len,
                    batch_size=batch_size * self.num_heads_kv,
                    dtype=dtype,
                    **kwargs,
                )
                # A shared norm owns its parameters once.
                if self.share_qk_norm:
                    keys = replace(keys, params=0, params_active=0)
                total += queries + keys
            elif self.norm_qk is not None:
                groups = (
                    (self.num_heads + self.num_heads_kv,)
                    if self.share_qk_norm
                    else (self.num_heads, self.num_heads_kv)
                )
                for head_rows in groups:
                    total += cost(
                        self.norm_qk,
                        seq_len=seq_len,
                        batch_size=batch_size * head_rows,
                        dtype=dtype,
                        **kwargs,
                    )
            if self.norm_out is not None:
                total += cost(
                    self.norm_out,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
            # ``Attention`` refuses a rotary embedding on a memory, so none runs.
            if self.rope is not None and memory_len < 0:
                total += cost(
                    self.rope,
                    seq_len=seq_len,
                    batch_size=1,
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
            return total

    def __init__(self, config: Config) -> None:
        _validate_head_dims(
            channels_in=config.channels_in,
            num_heads=config.num_heads,
            channels_head=config.channels_head,
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
            self.norm_q: TensorModule | None = None
            self.norm_k: TensorModule | None = None
        elif config.share_qk_norm:
            shared = config.norm_qk.make()
            self.norm_q = shared
            self.norm_k = shared
        else:
            self.norm_q = config.norm_qk.make()
            self.norm_k = config.norm_qk.make()
        self.norm_out: TensorModule | None = (
            config.norm_out.make() if config.norm_out else None
        )
        self.rope = config.rope.make() if config.rope else None

    def split_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project Q/K/V separately while reusing the fused parameter layout.

        Args:
          x: Input tokens shaped [..., sequence, channels_in].

        Returns:
          q: Queries shaped [..., sequence, num_heads, channels_head].
          k: Keys shaped [..., sequence, num_heads_kv, channels_head].
          v: Values with the same shape as k.

        """
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

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        self.proj_qkv.reset_parameters()
        self.proj_out.reset_parameters()
        if self.norm_q is not None:
            self.norm_q.reset_parameters()
        if self.norm_k is not None and self.norm_k is not self.norm_q:
            self.norm_k.reset_parameters()
        if self.norm_out is not None:
            self.norm_out.reset_parameters()
        if isinstance(self.rope, HasResetParameters):
            self.rope.reset_parameters()


# Only the inference contract, never a dimension's sign: torch raises on a negative
# extent when it builds the tensor, and re-checking here would add a second message for
# one fault (STYLE.md "Let the leaf complain").
def _validate_head_dims(
    *,
    channels_in: int,
    num_heads: int,
    channels_head: int,
) -> None:
    """Reject geometry that stayed unresolved after finalize."""
    if channels_head == -1 and channels_in != -1:
        raise ValueError(
            f"channels_in={channels_in} not divisible by "
            f"num_heads={num_heads}; set channels_head explicitly.",
        )
    if num_heads == -1 and channels_in != -1:
        raise ValueError(
            f"channels_in={channels_in} not divisible by "
            f"channels_head={channels_head}; set num_heads explicitly.",
        )
    if -1 in (channels_in, num_heads, channels_head):
        raise ValueError(
            f"Need at least two of channels_in={channels_in}, "
            f"num_heads={num_heads}, channels_head={channels_head}.",
        )


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


class Attention(AttentionProjections):
    """Multi-head attention with fused QKV and optional grouped-query heads.

    Self-attention by default. Given a ``memory``, it is cross-attention: the
    queries come from ``x`` and the keys and values from the memory, through
    the same ``proj_qkv`` (its first ``num_heads`` heads read ``x``, the rest
    the memory), so the memory has ``channels_in`` channels. Cross-attention
    is bidirectional over the memory and takes no window, rotary embedding or cache.
    The block that owns a cross-attention hands it the memory; a block whose
    attention is self-attention, as ``TransformerBlock``'s is, hands it none.
    """

    class Config(Makes["Attention"], AttentionProjections.Config, kw_only=False):
        _: KW_ONLY

        attn_kernel: Makeable[AttentionKernel] = field(default_factory=SdpaFused.Config)
        """Attention kernel shared by all heads."""

        @override
        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            memory_len: int = -1,
            **kwargs: object,
        ) -> Cost:
            """Add the kernel's products and, for self-attention, the per-token KV cache.

            The kernel is costed at the configured ``dropout`` over the whole
            sequence, or the whole memory: a window is a ``forward`` argument,
            not a config field, so the analytical reach is every key.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              memory_len: Memory positions per sequence the queries attend to;
                -1 attends to the sequence itself. A memory is attended
                without a cache, so it carries no state.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation cost over ``seq_len`` and ``batch_size``.

            """
            # Cross-attention prices ``seq_len`` queries against ``memory_len`` keys.
            reach = {} if memory_len < 0 else {"seq_len": memory_len, "rows": seq_len}
            kernel = cost(
                self.attn_kernel,
                **{"seq_len": seq_len} | reach,
                batch_size=batch_size,
                dtype=dtype,
                num_heads=self.num_heads,
                channels_head=self.channels_head,
                dropout_p=self.dropout,
                **kwargs,
            )
            return replace(
                super().cost(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    memory_len=memory_len,
                    **kwargs,
                )
                + kernel,
                bytes_state=0
                if memory_len >= 0
                else resolve_dtype(dtype).itemsize
                * 2
                * self.num_heads_kv
                * self.channels_head,
            )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
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
            raise RuntimeError(  # noqa: TRY004 -- The invalid sharding configuration is a runtime capability error, not a custom exception type.
                "Tensor parallelism requires a DTensor-compatible attention "
                "kernel; set attn_kernel=SdpaNaive (the fused flash kernel has "
                "no DTensor sharding strategy).",
            )

    def alloc_kv_cache(
        self,
        *,
        batch: int | tuple[int, ...],
        max_seq: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KVCache:
        """Allocate a KV cache sized for this attention block.

        Args:
          batch: Batch size (int) or multi-batch shape tuple.
          max_seq: Maximum sequence length the cache holds.
          device: Torch device placement; None defers to block's device.
          dtype: Tensor dtype; None defers to block's dtype.

        Returns:
          cache: Allocated KVCache instance for this head config.

        """
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
        memory: Tensor | None = None,
        cache: LayerCache | None = None,
        positions: Tensor | None = None,
        cos_sin: tuple[Tensor, Tensor] | None = None,
        dropout_p: float | None = None,
        is_causal: bool | None = None,
        attn_mask: Tensor | None = None,
        window: int = -1,
        **kwargs: object,
    ) -> Tensor:
        return self._forward(
            x,
            memory=memory,
            positions=positions,
            cos_sin=cos_sin,
            cache=cache,
            dropout_p=dropout_p,
            is_causal=is_causal,
            attn_mask=attn_mask,
            window=window,
            **kwargs,
        )

    def project_queries(self, x: Tensor) -> Tensor:
        """Project queries through the first ``num_heads`` heads of ``proj_qkv``.

        Projection only: ``norm_q`` is the caller's to apply.

        Args:
          x: Tokens ``[..., S, channels_in]``.

        Returns:
          q: Queries ``[..., S, num_heads, channels_head]``.

        """
        w = self.proj_qkv.weight.to(x.dtype)[: self.num_heads]
        q = torch.matmul(x, w.reshape(-1, w.shape[-1]).T)
        bias = self.proj_qkv.bias
        if bias is not None:
            q = q + bias.to(x.dtype)[: self.num_heads].reshape(-1)
        return q.reshape(*x.shape[:-1], *w.shape[:-1])

    def project_memory(self, memory: Tensor) -> tuple[Tensor, Tensor]:
        """Project keys and values through the remaining heads of ``proj_qkv``.

        Projection only: ``norm_k`` is the caller's to apply.

        Args:
          memory: Memory positions ``[..., M, channels_in]``.

        Returns:
          k: Keys ``[..., M, num_heads_kv, channels_head]``.
          v: Values, shaped like ``k``.

        """
        w = self.proj_qkv.weight.to(memory.dtype)[self.num_heads :]
        kv = torch.matmul(memory, w.reshape(-1, w.shape[-1]).T)
        bias = self.proj_qkv.bias
        if bias is not None:
            kv = kv + bias.to(memory.dtype)[self.num_heads :].reshape(-1)
        kv = kv.reshape(*memory.shape[:-1], *w.shape[:-1])
        k, v = kv.split([self.num_heads_kv, self.num_heads_kv], dim=-2)
        return k, v

    def _forward(
        self,
        x: Tensor,
        *,
        memory: Tensor | None = None,
        positions: Tensor | None,
        cos_sin: tuple[Tensor, Tensor] | None,
        cache: LayerCache | None,
        dropout_p: float | None,
        is_causal: bool | None,
        attn_mask: Tensor | None,
        window: int,
        **kwargs: object,
    ) -> Tensor:
        S = x.shape[-2]
        kv_cache: KVCache | None = None
        if cache is not None:
            state = cache[self.depth_index]
            assert isinstance(state, KVCache)
            kv_cache = state

        # proj_qkv: [..., S, C] -> [..., S, num_ensemble, channels_head].
        if memory is not None:
            # A block forwards its messages to every attention, so a window or a
            # causal call meant for self-attention reaches this one too.
            if (
                self.causal
                or is_causal
                or window != -1
                or self.rope is not None
                or cos_sin is not None
                or cache is not None
            ):
                raise ValueError(
                    "Attention to a memory takes no causal mask, window, rotary "
                    "embedding or cache.",
                )
            q = self.project_queries(x)
            k, v = self.project_memory(memory)
        elif self.split_qkv_projection:
            q, k, v = self.split_qkv(x)
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
                offset = kv_cache.seen if kv_cache is not None else 0
                positions = torch.arange(offset, offset + S, device=x.device)
            cos, sin = self.rope(positions)
            q, k = RoPE.rotate(q, k, cos, sin)

        # The cache stores [..., H, S, D]; the kernels take [..., S, H, D].
        q, k, v = (t.movedim(-3, -2) for t in (q, k, v))
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=-3)
            v = v.repeat_interleave(self.kv_groups, dim=-3)
        q, k, v = (t.movedim(-3, -2) for t in (q, k, v))

        # A multi-token chunk decoded against a longer cache must stay causal within the
        # chunk and honor ``window``; ``window_mask`` does both when finite, else
        # ``causal_chunk_mask`` gives the causal fallback. A single-token chunk needs
        # only ``window``, applied by the kernel's own fallback.
        causal = self.causal if is_causal is None else is_causal
        # A caller's mask keeps the flag, so the kernel folds causality into it.
        flag = causal and (k.shape[-3] == S or attn_mask is not None)
        if attn_mask is None and causal and S > 1:
            attn_mask = window_mask(q, k, window=window)
            if attn_mask is None:
                attn_mask = causal_chunk_mask(q, k)

        out = self.attn_kernel(
            q,
            k,
            v,
            dropout_p=(
                (self.dropout if self.training else 0.0)
                if dropout_p is None
                else dropout_p
            ),
            is_causal=flag,
            attn_mask=attn_mask,
            window=window,
            **kwargs,
        )

        # [..., S, H, D] -> [..., S, H*D].
        out = out.flatten(-2)
        if self.norm_out is not None:
            out = self.norm_out(out)
        return self.proj_out(out)
