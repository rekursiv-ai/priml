"""Joint attention over multiple token streams."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, field
from typing import Self, cast, override

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.distributed.tensor import DTensor

import torch

from priml.model.attention.kernel import SdpaFused
from priml.model.attention.kvcache import KVCache
from priml.model.attention.rope import RoPE
from priml.model.attention.self_attention import (
    AttentionProjections,
    _infer_head_dims,
    _validate_head_dims,
)
from priml.model.attention.window import causal_chunk_mask
from priml.model.custom_types import (
    AttentionKernel,
    ChannelsIn,
    DepthIndex,
    RotaryFactors,
)
from priml.model.init import InitFn, kaiming_uniform
from priml.model.linear import EnsembleLinear, Linear


class MultiStreamAttention(nn.Module):
    """N-stream joint attention with per-stream QKV and shared K/V.

    List-form of SelfAttention: each config field and forward kwarg
    mirrors SA but accepts one value per stream. K/V from all streams
    are concatenated so each stream's Q attends to the full set.
    """

    class Config(Fig["MultiStreamAttention"], kw_only=False):
        channels_in: int = -1
        """Channel width shared across all streams."""

        channels_out: int = -1
        """Number of output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        num_streams: int = 2
        """Number of parallel token streams when streams is empty."""

        streams: list[AttentionProjections.Config] = field(
            default_factory=list[AttentionProjections.Config],
        )
        """Explicit stream subtrees.

        Residual/head geometry and depth come from the parent.
        Each stream owns its projections, norms, RoPE, dropout and causal policy.
        An empty list retains the shared norm configuration below.
        """

        num_heads: int = 8
        """Query-head count."""

        channels_head: int = -1
        """Per-head dimension (-1 = channels_in // num_heads)."""

        num_heads_kv: int = -1
        """Key/value-head count for GQA (-1 = same as num_heads)."""

        bias: bool = False
        """Bias in QKV and output projections."""

        dropout: float = 0.0
        """Attention dropout probability."""

        causal: bool = False
        """Apply causal (autoregressive) attention mask."""

        rope: list[Makeable[RotaryFactors] | None] = field(
            default_factory=list["Makeable[RotaryFactors] | None"],
        )
        """Per-stream rotary embeddings (empty = no internal RoPE)."""

        norm_qk: Makeable[nn.Module] | None = None
        """Optional norm applied to Q and K before attention."""

        share_qk_norm: bool = True
        """Reuse one ``norm_qk`` instance for both Q and K.

        When False, two independent modules are built from the same
        config (shared across streams, but Q/K-independent).
        """

        norm_out: Makeable[nn.Module] | None = None
        """Optional norm applied to attention output before proj_out."""

        attn_kernel: Makeable[AttentionKernel] = field(
            default_factory=SdpaFused.Config,
        )
        """Attention kernel (SdpaFused or SdpaNaive)."""

        init_weight: InitFn = kaiming_uniform
        """Weight initialization function."""

        depth_index: DepthIndex = ()
        """Block depth index for depth-scaled init."""

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
            if self.streams:
                self.num_streams = len(self.streams)
                for stream in self.streams:
                    stream.channels_in = self.channels_in
                    stream.channels_out = self.channels_out
                    stream.num_heads = self.num_heads
                    stream.num_heads_kv = self.num_heads_kv
                    stream.channels_head = self.channels_head
                    stream.depth_index = self.depth_index
            return super().finalize()

    def __init__(self, config: Config) -> None:
        _validate_head_dims(
            channels_in=config.channels_in,
            num_heads=config.num_heads,
            channels_head=config.channels_head,
        )
        if (
            -1 not in (config.channels_in, config.channels_out)
            and config.channels_in != config.channels_out
        ):
            raise ValueError(
                f"channels_in={config.channels_in} must equal "
                f"channels_out={config.channels_out} for {type(self).__name__}.",
            )
        super().__init__()
        if config.num_heads % config.num_heads_kv != 0:
            raise ValueError(
                f"num_heads={config.num_heads} must be divisible by "
                f"num_heads_kv={config.num_heads_kv}.",
            )
        if config.causal and config.num_streams > 1:
            raise ValueError("causal=True requires num_streams=1.")
        self.num_streams = config.num_streams
        self.num_heads = config.num_heads
        self.channels_head = config.channels_head
        self.num_heads_kv = config.num_heads_kv
        self.kv_groups = config.num_heads // config.num_heads_kv
        self.dropout = config.dropout
        self.causal = config.causal
        self.depth_index = config.depth_index
        self.streams: nn.ModuleList[AttentionProjections] = nn.ModuleList()
        if config.streams:
            self._init_streams(config)
            return

        c = config.channels_in
        N = config.num_streams
        # Residual width (c) decoupled from attention inner width.
        inner = config.num_heads * config.channels_head

        self.proj_qkvs = nn.ModuleList(
            EnsembleLinear.Config(
                channels_in=c,
                channels_out=config.channels_head,
                num_ensemble=config.num_heads + 2 * config.num_heads_kv,
                bias=config.bias,
                depth_index=config.depth_index,
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
                depth_index=config.depth_index,
                init_weight=config.init_weight,
                shard="rowwise",
            ).make()
            for _ in range(N)
        )

        # Per-stream RoPE keyed by stream index; absent = no RoPE.
        # ``ModuleDict`` registers submodules, so what a slot BUILDS must be an
        # ``nn.Module`` even though the slot is typed by what it is CALLED as.
        self.ropes = nn.ModuleDict(
            {
                str(i): cast(nn.Module, cfg.make())
                for i, cfg in enumerate(config.rope)
                if cfg is not None
            },
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
        projections = (
            [stream.proj_qkv for stream in self.streams]
            if self.streams
            else self.proj_qkvs
        )
        sharded = any(isinstance(qkv.weight, DTensor) for qkv in projections)
        if isinstance(self.attn_kernel, SdpaFused) and sharded:
            raise RuntimeError(
                "Tensor parallelism requires a DTensor-compatible attention "
                "kernel; set attn_kernel=SdpaNaive (the fused flash kernel has "
                "no DTensor sharding strategy).",
            )

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        if self.streams:
            for stream in self.streams:
                stream.reset_parameters()
            return
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

    def load_stream(self, index: int, *, source: AttentionProjections) -> None:
        """Copy native attention weights into one explicit stream.

        The caller must configure matching non-parameter semantics (normalization
        epsilon, rotary frequencies, dropout and masking). This loads weights,
        not architecture. All keys and shapes are validated before any copy.

        Args:
          index: Explicit destination stream index.
          source: Native attention with fused QKV parameters.

        """
        if not self.streams:
            raise ValueError("Native loading requires explicit streams.")
        if index < 0 or index >= len(self.streams):
            raise ValueError(f"Invalid stream index {index}.")
        target = self.streams[index]
        _validate_native_state(target, source=source)
        target.load_state_dict(source.state_dict(), strict=True)

    @override
    def forward(
        self,
        xs: Sequence[Tensor],
        *,
        positions: Sequence[Tensor | list[Tensor] | None] | None = None,
        cos_sin: Sequence[tuple[Tensor, Tensor] | None] | None = None,
        dropout_p: float | None = None,
        is_causal: bool | None = None,
        attn_mask: Tensor | Sequence[Tensor | None] | None = None,
        **kwargs: object,
    ) -> tuple[Tensor, ...]:
        """Attend jointly without caching.

        Args:
          xs: Token streams shaped [..., S_i, channels_in].
          positions: Optional position indices for each stream.
          cos_sin: Optional rotary factors for each stream.
          dropout_p: Override the configured attention dropout probability.
          is_causal: Override the configured causal policy.
          attn_mask: One broadcast mask, or one mask per query stream. Per-stream
            masks broadcast to [..., heads, S_i, sum(S_j)]; None is unrestricted.
          **kwargs: Open message bus forwarded to the shared kernel.

        Returns:
          outputs: One tensor per input stream, with the original shape.

        """
        outputs, caches = self._forward(
            xs,
            positions=positions,
            cos_sin=cos_sin,
            cache=None,
            dropout_p=dropout_p,
            is_causal=is_causal,
            attn_mask=attn_mask,
            **kwargs,
        )
        assert caches is None
        return outputs

    def forward_cached(
        self,
        xs: Sequence[Tensor],
        *,
        cache: Sequence[KVCache | None],
        positions: Sequence[Tensor | list[Tensor] | None] | None = None,
        cos_sin: Sequence[tuple[Tensor, Tensor] | None] | None = None,
        dropout_p: float | None = None,
        is_causal: bool | None = None,
        attn_mask: Tensor | Sequence[Tensor | None] | None = None,
        **kwargs: object,
    ) -> tuple[tuple[Tensor, ...], list[KVCache]]:
        """Attend jointly using and updating per-stream caches.

        Args:
          xs: Input tensors, one per stream.
          cache: Per-stream KVCache or None for prefill.
          positions: Per-stream position indices.
          cos_sin: Per-stream pre-computed sin/cos for rotary embeddings.
          dropout_p: Attention dropout rate (None for inference).
          is_causal: Whether to apply causal mask.
          attn_mask: Per-stream attention mask or single mask for all.
          **kwargs: Extra arguments forwarded to the base forward.

        Returns:
          result: Tuple of per-stream outputs and updated KVCaches.

        """
        outputs, updated = self._forward(
            xs,
            positions=positions,
            cos_sin=cos_sin,
            cache=cache,
            dropout_p=dropout_p,
            is_causal=is_causal,
            attn_mask=attn_mask,
            **kwargs,
        )
        assert updated is not None
        return outputs, updated

    def _forward(
        self,
        xs: Sequence[Tensor],
        *,
        positions: Sequence[Tensor | list[Tensor] | None] | None,
        cos_sin: Sequence[tuple[Tensor, Tensor] | None] | None,
        cache: Sequence[KVCache | None] | None,
        dropout_p: float | None,
        is_causal: bool | None,
        attn_mask: Tensor | Sequence[Tensor | None] | None,
        **kwargs: object,
    ) -> tuple[tuple[Tensor, ...], list[KVCache] | None]:
        N = self.num_streams
        if len(xs) != N:
            raise ValueError(f"Expected {N} streams, got {len(xs)}.")
        masks = list(attn_mask) if isinstance(attn_mask, Sequence) else [attn_mask] * N
        for name, values in (
            ("attn_mask", masks),
            ("positions", positions),
            ("cos_sin", cos_sin),
            ("cache", cache),
        ):
            if values is not None and len(values) != N:
                raise ValueError(f"{name} must contain one entry per stream ({N}).")

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
        out_caches: list[KVCache] = []

        for i, x in enumerate(xs):
            S = x.shape[-2]
            stream = self.streams[i] if self.streams else None
            projection = stream.proj_qkv if stream is not None else self.proj_qkvs[i]
            if stream is not None and stream.split_qkv_projection:
                q, k, v = stream.split_qkv(x)
            else:
                q, k, v = (
                    t.contiguous()
                    for t in projection(x).split(
                        [self.num_heads, self.num_heads_kv, self.num_heads_kv],
                        dim=-2,
                    )
                )

            norm_q = stream.norm_q if stream is not None else self.norm_q
            if norm_q is not None:
                q = norm_q(q)
            norm_k = stream.norm_k if stream is not None else self.norm_k
            if norm_k is not None:
                k = norm_k(k)

            # RoPE: prefer external cos_sin, fall back to internal rope.
            cs_i = cs_list[i]
            si = str(i)
            rope = None
            if stream is not None:
                rope = stream.rope
            elif si in self.ropes:
                rope = self.ropes[si]
            if cs_i is not None:
                cos, sin = cs_i
                q, k = RoPE.rotate(q, k, cos, sin)
            elif rope is not None:
                p = pos_list[i]
                if p is None:
                    c_i = cache_list[i]
                    offset = c_i.seen if c_i is not None else 0
                    p = torch.arange(offset, offset + S, device=x.device)
                assert isinstance(rope, nn.Module)
                cos, sin = rope(p)
                q, k = RoPE.rotate(q, k, cos, sin)

            # The cache stores [..., H, S, hd]; the kernels take [..., S, H, hd].
            q, k, v = (t.movedim(-3, -2) for t in (q, k, v))
            c_i = cache_list[i]
            if cache is not None:
                if c_i is not None:
                    k, v = c_i.update(k, v)
                else:
                    c_i = KVCache(k, v)
                out_caches.append(c_i)

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
        k_cat, v_cat = (t.movedim(-3, -2) for t in (k_cat, v_cat))
        results: list[Tensor] = []
        for i in range(N):
            q_i = all_q[i].movedim(-3, -2)
            S_i = q_i.shape[-3]
            stream = self.streams[i] if self.streams else None
            causal = stream.causal if stream is not None else self.causal
            dropout = stream.dropout if stream is not None else self.dropout
            mask = masks[i]
            out = self.attn_kernel(
                q_i,
                k_cat,
                v_cat,
                dropout_p=(
                    (dropout if self.training else 0.0)
                    if dropout_p is None
                    else dropout_p
                ),
                is_causal=(
                    causal and total_kv == S_i if is_causal is None else is_causal
                ),
                attn_mask=(
                    causal_chunk_mask(q_i, k_cat) if mask is None and causal else mask
                ),
                **kwargs,
            )
            # [..., S, H, hd] → [..., S, H*hd].
            out = out.flatten(-2)
            norm_out = stream.norm_out if stream is not None else self.norm_out
            if norm_out is not None:
                out = norm_out(out)
            projection_out = (
                stream.proj_out if stream is not None else self.proj_outs[i]
            )
            out = projection_out(out)
            results.append(out)

        outputs = tuple(results)
        if cache is None:
            return outputs, None
        return outputs, out_caches

    def _init_streams(self, config: Config) -> None:
        if len(config.streams) > 1 and any(s.causal for s in config.streams):
            raise ValueError(
                "Causal streams require a single stream; use per-stream masks.",
            )
        self.streams = nn.ModuleList(
            AttentionProjections(stream) for stream in config.streams
        )
        self.norm_q = self.norm_k = self.norm_out = None
        self.attn_kernel = config.attn_kernel.make()


def _validate_native_state(target: nn.Module, *, source: nn.Module) -> None:
    """Validate the entire native transfer before any destination tensor changes."""
    expected, actual = target.state_dict(), source.state_dict()
    if expected.keys() != actual.keys():
        raise ValueError(
            "Native stream state keys do not match the configured destination.",
        )
    for name, tensor in expected.items():
        if tensor.shape != actual[name].shape:
            raise ValueError(f"Native stream state shape mismatch for {name}.")
