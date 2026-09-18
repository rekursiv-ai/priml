"""Multi-head Latent Attention (MLA).

Used by DeepSeek-V2/V3 and Kimi-K2. Compresses KV through a low-rank
``kv_lora_rank`` latent, applies RoPE only on a decoupled
``channels_qk_rope_head`` slice, and uses a separate ``channels_v_head`` for the
value path.

Q has an optional LoRA decomposition (``q_lora_rank``): DeepSeek-V3 uses
it (1536), Kimi-K2 does not (None ⇒ single ``proj_q``).

Forward shape map. B batch · S new tokens · T cached+new · n ``num_heads`` ·
D ``channels_qk_nope_head`` · R ``channels_qk_rope_head`` ·
V ``channels_v_head`` · L ``kv_lora_rank``::

    x [B, S, hidden]
    │
    ├─ proj_q ⟶ q [B, S, n, D+R]          (proj_q_a ⟶ norm ⟶ proj_q_b
    │  │                                    when q_lora_rank is set)
    │  ├─ q_nope [B, S, n, D]
    │  └─ q_pe   [B, S, n, R] ⟵ RoPE
    │
    └─ proj_kv_a ⟶ [B, S, L+R]
       ├─ c_kv [B, S, L] ⟵ kv_a_layernorm  ╮ cached; head-shared,
       └─ k_pe [B, S, R] ⟵ RoPE            ╯ so no head axis

Attention never materializes K or V. ``proj_kv_b.weight`` is viewed per head
as ``W_KR [n, D, L]`` and ``W_UV [n, V, L]``, then folded into the contraction
so both terms meet in the latent space::

    logits = (q_nope @ W_KR) @ c_kv^T + q_pe @ k_pe^T   [B, n, S, T]
    out    = (softmax(logits) @ c_kv) @ W_UV            [B, S, n, V]
                                            ⟶ proj_out ⟶ [B, S, hidden]

**Cache layout.** Only ``(c_kv, k_pe)`` are cached -- not the expanded
K/V. For Kimi-K2 (n=64, D+V=256, L=512, R=64) this cuts cache memory
~25× (576 dims/token vs 16384).

**Absorb-math cost.** Decode avoids expanding the cached latent into
per-head K/V tensors before attention, saving that projection on every step.
Numerical output is identical up to softmax
ordering; verified against re-expand in the test suite.

Both are :class:`LatentAttention` under its ``absorb`` flag, because
``(q @ W_KR) @ c_kv`` and ``q @ (W_KR @ c_kv)`` are the same product: they
differ only in where the latent projections are applied, and so in what the
kernel's K and V are. Everything after -- mask, softmax, dropout -- is the
ordinary :class:`~priml.model.custom_types.AttentionKernel` every other
attention in priml uses.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field, fields, replace
from functools import partial
from typing import TYPE_CHECKING, Literal, Self, override

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    parallelize_module,
)

import torch

from priml.cost import (
    Cost,
    cost,
    resolve_dtype,
    traffic,
)
from priml.model.attention.kernel import (
    SdpaFused,
    SdpaNaive,
)
from priml.model.attention.kvcache import KVCache
from priml.model.attention.rope import RoPE, rotation_cost
from priml.model.attention.window import causal_chunk_mask
from priml.model.custom_types import (
    AttentionKernel,
    ChannelsIn,
    ChannelsInOutConfig,
    ChannelsOut,
    DepthIndex,
    HasResetParameters,
    LatentAttentionKernel,
    RotaryConfig,
    TensorModule,
    WeightedTensorModule,
    infer_same_width,
)
from priml.model.init import InitFn, kaiming_uniform
from priml.model.legacy_keys import absorb_legacy_keys
from priml.model.linear import Linear
from priml.model.norm import RMSNorm


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


class LatentAttention(nn.Module):
    """MLA attention as ONE contraction, associated one of two ways.

    ``(q_nope @ W_KR) @ c_kv`` and ``q_nope @ (W_KR @ c_kv)`` are the same
    product, so absorb and re-expand are not two attention algorithms -- they
    are two parenthesizations of one. What differs is only WHERE the latent
    projections are applied, and therefore what the kernel's K and V are:

    - ``absorb=True`` folds ``W_KR`` into the query and ``W_UV`` into the
      output, so K and V are the ``kv_lora_rank`` latent itself, head-shared
      and broadcast. Nothing is ever materialized at ``channels_v_head``
      width, which is the ~25x cache saving MLA exists for.
    - ``absorb=False`` applies both to the latent first, so K and V are the
      ordinary per-head tensors. It costs the saving and buys nothing here --
      it exists because it is the form the literature states, so the two being
      equal is a claim worth testing rather than asserting.

    Either way the mask, softmax, dropout and the ``[..., S, num_heads, ...]``
    layout belong to the injected :class:`AttentionKernel`, spelled once
    for every attention in priml rather than again per MLA variant.
    """

    class Config(Fig["LatentAttention"]):
        absorb: bool = True
        """Fold the latent projections into the query and output.

        The default, because not materializing K/V is MLA's reason to exist.
        Turn it off to attend over expanded K/V -- which is what a fused
        kernel needs, since the absorbed form's K is the latent."""

        attn_kernel: Makeable[AttentionKernel] = field(
            default_factory=SdpaNaive.Config,
        )
        """Kernel the contraction is handed to.

        ``SdpaNaive`` rather than the fused default the rest of priml takes:
        under ``absorb`` the value is a head-shared BROADCAST view, and a
        fused kernel materializes it -- which spends exactly the memory the
        absorbed form was avoiding."""

        def cost(
            self,
            *,
            seq_len: int,
            dtype: torch.dtype | None,
            num_heads: int,
            channels_head: int,
            channels_v_head: int,
            kv_lora_rank: int,
            channels_qk_rope_head: int,
            dropout_p: float,
        ) -> Cost:
            """Cost the kernel's two products at the widths this form hands it.

            Absorbed, K is the latent plus the rope key and V the latent alone;
            re-expanded, they are the per-head Q/K width and ``channels_v_head``.
            The owner costs projection FLOPs and weights as ``proj_kv_b``.
            Absorption additionally moves two per-head latent rows where that
            dense projection moves one shared row; charge that operand delta
            here because this config owns the association choice.

            The kernel receives separate query/key and value widths so each
            product's operands are costed directly, without scaling unrelated
            score or softmax traffic. This is not ``HasCost``: a kernel config
            holds no shapes, so the owner states them, the way it hands the
            latent to ``forward``.

            Args:
              seq_len: Tokens per sequence.
              dtype: Activation dtype; ``None`` is torch's default.
              num_heads: Query heads.
              channels_head: Width of each query/key head.
              channels_v_head: Width of each value head.
              kv_lora_rank: Width of the shared KV latent.
              channels_qk_rope_head: Width of the rotated key slice.
              dropout_p: Attention dropout rate.

            Returns:
              cost: Per-token cost of this module.

            """
            if self.absorb:
                channels_k = kv_lora_rank + channels_qk_rope_head
                channels_v = kv_lora_rank
            else:
                channels_k, channels_v = channels_head, channels_v_head
            kernel = cost(
                self.attn_kernel,
                seq_len=seq_len,
                dtype=dtype,
                num_heads=num_heads,
                channels_head=channels_k,
                channels_v_head=channels_v,
                dropout_p=dropout_p,
            )
            moved = (2 * num_heads - 1) * kv_lora_rank if self.absorb else 0
            dt = dtype
            return (
                kernel
                + traffic("primal", "matmul", elements=moved, dtype=dt)
                + traffic("adjoint", "matmul", elements=2 * moved, dtype=dt)
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.absorb = config.absorb
        self.attn_kernel = config.attn_kernel.make()

    @override
    def forward(
        self,
        q_nope: Tensor,
        q_pe: Tensor,
        c_kv: Tensor,
        k_pe: Tensor,
        *,
        w_kr: Tensor,
        w_uv: Tensor,
        **kwargs: object,
    ) -> Tensor:
        num_heads = q_nope.shape[-2]
        if self.absorb:
            # Latent as K and V. Both are head-SHARED, so they broadcast over
            # the head axis instead of being expanded per head.
            q = torch.einsum("...shd,hdl->...shl", q_nope, w_kr)
            k, v = _broadcast_heads(c_kv, num_heads), _broadcast_heads(c_kv, num_heads)
        else:
            q = q_nope
            k = torch.einsum("...tl,hdl->...thd", c_kv, w_kr)
            v = torch.einsum("...tl,hvl->...thv", c_kv, w_uv)

        # One contraction over the concatenated widths: a dot product over
        # concatenated vectors is the sum of the pieces' dot products, so the
        # nope and rope logit terms need not be computed apart and added.
        # ``k_pe`` is head-shared like the latent.
        out = self.attn_kernel(
            torch.cat([q, q_pe], dim=-1),
            torch.cat([k, _broadcast_heads(k_pe, num_heads)], dim=-1),
            v,
            **kwargs,
        )
        # Absorbed output is still in latent space; project it to the value
        # width. The re-expand path applied ``W_UV`` to ``v`` already.
        return torch.einsum("...shl,hvl->...shv", out, w_uv) if self.absorb else out


def _broadcast_heads(x: Tensor, num_heads: int) -> Tensor:
    """View a head-shared tensor as per-head, without copying it."""
    return x.unsqueeze(-2).expand(*x.shape[:-1], num_heads, x.shape[-1])


class MultiHeadLatentAttention(nn.Module):
    """MLA with decoupled RoPE and compressed KV latent."""

    class Config(Fig["MultiHeadLatentAttention"], kw_only=False):
        channels_in: int = -1
        """Input channel width."""

        channels_out: int = -1
        """Output channel width."""

        _: KW_ONLY

        num_heads: int = -1
        """Attention-head count shared by Q, K, and V."""

        channels_qk_nope_head: int = 128
        """Non-RoPE portion of the Q/K head dim."""

        channels_qk_rope_head: int = 64
        """RoPE portion of the Q/K head dim (shared key across heads)."""

        channels_v_head: int = 128
        """Value head dim (independent of Q/K head dim)."""

        q_lora_rank: int | None = None
        """Rank for Q LoRA. ``None`` ⇒ single ``proj_q`` (Kimi-K2). Non-None
        uses ``proj_q_a`` + RMSNorm + ``proj_q_b`` (DeepSeek-V3)."""

        kv_lora_rank: int = 512
        """Rank of the compressed KV latent."""

        bias: bool = False
        """Include bias in projections. MLA typically has bias=False."""

        dropout: float = 0.0
        """Attention dropout probability."""

        causal: bool = True
        """Apply causal mask."""

        softmax_scale: float | None = None
        """Override the SDPA scale. Default ``1/sqrt(qk_nope + qk_rope)``."""

        rope: RotaryConfig | None = None
        """Rotary embedding applied to the ``qk_rope`` slice. ``None`` ⇒ none.

        Typed by what MLA CALLS -- positions in, ``(cos, sin)`` out -- rather
        than by ``RoPE``, so a learned or NTK-scaled variant fills the slot."""

        attn_kernel: Makeable[LatentAttentionKernel] = field(
            default_factory=LatentAttention.Config,
        )
        """How the latent is attended.

        Typed by the contract rather than by :class:`LatentAttention`, so a
        kernel this module never saw fills the slot by implementing it."""

        norm_q_lora: Makeable[TensorModule] = field(
            default_factory=partial(RMSNorm.Config, elementwise_affine=True),
        )
        """Normalization between the Q LoRA's two projections.

        Built only when ``q_lora_rank`` is set; its width is the rank."""

        proj_q: Makeable[TensorModule] = field(default_factory=Linear.Config)
        """Q projection: model width in, ``num_heads * channels_qk_head`` out.

        Built only when ``q_lora_rank`` is None; the LoRA path uses
        ``proj_q_a``/``proj_q_b`` instead. Sharded head-parallel under tensor
        parallelism (see :meth:`tensor_parallel_plan`)."""

        proj_q_a: Makeable[TensorModule] = field(default_factory=Linear.Config)
        """Q LoRA down-projection to ``q_lora_rank``. Replicated: the rank is
        head-shared, so sharding it over the head dim is a correctness bug."""

        proj_q_b: Makeable[TensorModule] = field(default_factory=Linear.Config)
        """Q LoRA up-projection to ``num_heads * channels_qk_head``, head-major.
        Sharded head-parallel, like ``proj_q``."""

        proj_kv_a: Makeable[TensorModule] = field(default_factory=Linear.Config)
        """KV down-projection to ``kv_lora_rank + channels_qk_rope_head``.
        Replicated -- this is the latent the cache holds."""

        proj_kv_b: ChannelsInOutConfig[WeightedTensorModule] = field(
            default_factory=Linear.Config,
        )
        """KV latent expansion. Replicated even under tensor parallelism: the
        absorb math slices its weight per head in Python, so each rank holds the
        whole thing and expands only its own head range."""

        proj_out: Makeable[TensorModule] = field(default_factory=Linear.Config)
        """Output projection back to model width; rowwise-sharded.

        Typed by its CALL shape, unlike the others: this is the one projection
        whose result is returned directly, so a bare ``nn.Module`` (whose
        ``__call__`` is untyped) would make the forward's return ``Any``."""

        norm_kv_lora: Makeable[TensorModule] = field(
            default_factory=partial(RMSNorm.Config, elementwise_affine=True),
        )
        """Normalization on the compressed KV latent; its width is
        ``kv_lora_rank``."""

        init_weight: InitFn = kaiming_uniform
        """Weight init for linear projections."""

        depth_index: DepthIndex = ()
        """Block depth for depth-scaled init (-1 = no scaling)."""

        shard: Literal["colwise"] | None = None
        """Tensor-parallel shard style over the mesh tp dim; ``None`` replicates.

        ``"colwise"`` selects the head-parallel custom style (see
        :meth:`MultiHeadLatentAttention.tensor_parallel_style`): the q-path and
        ``proj_out`` shard over the head dim while the head-shared latent path
        stays replicated. MLA exposes no ``"rowwise"`` -- it is sharded as one
        block, not per child projection.
        """

        @property
        def channels_qk_head(self) -> int:
            """Channels qk head."""
            return self.channels_qk_nope_head + self.channels_qk_rope_head

        @property
        def channels_head(self) -> int:
            """The Q/K head width, which is what rotary factors are sized to.

            NOT ``channels_v_head``: MLA's value path is independently sized, and
            only the query/key width participates in the rotation.
            """
            return self.channels_qk_head

        @override
        def finalize(self) -> Self:
            infer_same_width(self)
            if (
                self.q_lora_rank is not None
                and isinstance(self.norm_q_lora, ChannelsIn)
                and self.norm_q_lora.channels_in == -1
            ):
                self.norm_q_lora.channels_in = self.q_lora_rank
            if (
                isinstance(self.norm_kv_lora, ChannelsIn)
                and self.norm_kv_lora.channels_in == -1
            ):
                self.norm_kv_lora.channels_in = self.kv_lora_rank
            self._size_projections()
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the projections, norms, rotary, kernel, and the latent cache.

            ``bytes_state`` is what :meth:`alloc_kv_cache` stores per token: the
            ``kv_lora_rank`` latent and the head-shared rope key, not the K/V the
            kernel would expand them into.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            Raises:
              TypeError: The kernel slot's config carries no ``cost``.

            """
            q_path = (
                (self.proj_q,)
                if self.q_lora_rank is None
                else (self.proj_q_a, self.norm_q_lora, self.proj_q_b)
            )
            children = (
                *q_path,
                self.proj_kv_a,
                self.norm_kv_lora,
                self.proj_kv_b,
                self.proj_out,
            )
            total = sum(
                (
                    cost(
                        child,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        dtype=dtype,
                        **kwargs,
                    )
                    for child in children
                ),
                Cost(),
            )
            if self.rope is not None:
                total += cost(
                    self.rope,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                # Every query head and the one shared key row are rotated.
                total += rotation_cost(
                    self.rope,
                    rows=seq_len * batch_size,
                    dtype=dtype,
                    channels_head=self.channels_qk_rope_head,
                    heads=self.num_heads + 1,
                )
            total += cost(
                self.attn_kernel,
                seq_len=seq_len,
                dtype=dtype,
                num_heads=self.num_heads,
                channels_head=self.channels_qk_head,
                channels_v_head=self.channels_v_head,
                kv_lora_rank=self.kv_lora_rank,
                channels_qk_rope_head=self.channels_qk_rope_head,
                dropout_p=self.dropout,
            )
            return replace(
                total,
                bytes_state=resolve_dtype(dtype).itemsize
                * (self.kv_lora_rank + self.channels_qk_rope_head),
            )

        # Every width here is DERIVED -- a head count times a per-head width, or a LoRA
        # rank -- so a caller states the shape once and swaps the projection class
        # without restating any of it. Only the sentinel fields are filled: a slot
        # carrying a width, a bias, or an init the caller chose deliberately is left
        # alone, since overwriting it would build a model that differs from the
        # configured one.
        def _size_projections(self) -> None:
            """Fill each projection slot's widths from the shape fields."""
            qk_out = self.num_heads * self.channels_qk_head
            # ``object`` because the slots differ in what they BUILD (a plain
            # module, or one whose call shape is named); this loop only sets
            # widths, which it gates on a Protocol either way.
            widths: list[tuple[object, int, int]] = [
                (
                    self.proj_kv_a,
                    self.channels_in,
                    self.kv_lora_rank + self.channels_qk_rope_head,
                ),
                (
                    self.proj_kv_b,
                    self.kv_lora_rank,
                    self.num_heads
                    * (self.channels_qk_nope_head + self.channels_v_head),
                ),
                (
                    self.proj_out,
                    self.num_heads * self.channels_v_head,
                    self.channels_in,
                ),
            ]
            if self.q_lora_rank is None:
                widths.append((self.proj_q, self.channels_in, qk_out))
            else:
                widths.append((self.proj_q_a, self.channels_in, self.q_lora_rank))
                widths.append((self.proj_q_b, self.q_lora_rank, qk_out))
            for slot, c_in, c_out in widths:
                if isinstance(slot, ChannelsIn) and slot.channels_in == -1:
                    slot.channels_in = c_in
                if isinstance(slot, ChannelsOut) and slot.channels_out == -1:
                    slot.channels_out = c_out
                if isinstance(slot, Linear.Config):
                    _fill_unset(slot, "bias", self.bias)
                    _fill_unset(slot, "init_weight", self.init_weight)
                    if not slot.depth_index:
                        slot.depth_index = self.depth_index

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.dropout < 0.0 or config.dropout > 1.0:
            raise ValueError(f"dropout must be between 0 and 1, got {config.dropout}.")
        self.num_heads = config.num_heads
        self.channels_qk_nope_head = config.channels_qk_nope_head
        self.channels_qk_rope_head = config.channels_qk_rope_head
        self.channels_qk_head = (
            config.channels_qk_nope_head + config.channels_qk_rope_head
        )
        self.channels_v_head = config.channels_v_head
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.dropout = config.dropout
        self.causal = config.causal
        self.depth_index = config.depth_index
        # ``is None``, not truthiness: 0.0 is a valid scale.
        self.softmax_scale = (
            self.channels_qk_head**-0.5
            if config.softmax_scale is None
            else config.softmax_scale
        )
        self.shard = config.shard

        # Tensor parallelism is head-parallel: the custom ``ParallelStyle``
        # (see ``tensor_parallel_style``) shards the q-path and ``proj_out`` over
        # the head dim and makes the absorb-math rank-local. Until that style is
        # applied these record the replicated identity (every rank owns all
        # num_heads); the head-shared latent path always stays replicated.
        self._tp_mesh: DeviceMesh | None = None
        self._heads_local = config.num_heads
        self._head_offset = 0

        # Q path. The attribute names below are the CONTRACT the tensor-parallel
        # plan names (see ``tensor_parallel_plan``); the slots they are built
        # from are the caller's choice.
        if config.q_lora_rank is None:
            self.proj_q = config.proj_q.make()
            self.proj_q_a: TensorModule | None = None
            self.q_a_layernorm: TensorModule | None = None
            self.proj_q_b: TensorModule | None = None
        else:
            self.proj_q = None
            self.proj_q_a = config.proj_q_a.make()
            self.q_a_layernorm = config.norm_q_lora.make()
            self.proj_q_b = config.proj_q_b.make()

        # KV path: compressed_kv is split into (c_kv, k_pe).
        self.proj_kv_a = config.proj_kv_a.make()
        self.kv_a_layernorm = config.norm_kv_lora.make()
        # kv_b expands c_kv into (n_heads * (qk_nope + channels_v_head)).
        self.proj_kv_b = config.proj_kv_b.make()
        self.proj_out = config.proj_out.make()

        self.rope = config.rope.make() if config.rope else None
        self.attn_kernel = config.attn_kernel.make()
        absorb_legacy_keys(
            self,
            {
                "q_proj": "proj_q",
                "q_a_proj": "proj_q_a",
                "q_b_proj": "proj_q_b",
                "kv_a_proj": "proj_kv_a",
                "kv_b_proj": "proj_kv_b",
                "o_proj": "proj_out",
            },
        )

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        for m in (
            self.proj_q,
            self.proj_q_a,
            self.q_a_layernorm,
            self.proj_q_b,
            self.proj_kv_a,
            self.kv_a_layernorm,
            self.proj_kv_b,
            self.proj_out,
            self.rope,
            self.attn_kernel,
        ):
            if isinstance(m, HasResetParameters):
                m.reset_parameters()

    def alloc_kv_cache(
        self,
        *,
        batch: int | tuple[int, ...],
        max_seq: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KVCache:
        """Allocate a compressed-latent cache for this attention block.

        ``KVCache.k`` stores the per-token ``c_kv`` latent;
        ``KVCache.v`` stores the per-token ``k_pe``. Both have a
        single "head" axis of size 1 (the latent is head-shared).

        Args:
          batch: Batch size or shape tuple (supports dynamic batch).
          max_seq: Maximum sequence length to allocate storage for.
          device: Torch device (cuda/cpu); defaults to model device.
          dtype: Tensor dtype; defaults to model dtype.

        Returns:
          cache: Allocated KVCache with k:[batch, 1, max_seq, kv_lora_rank].

        """
        return KVCache.alloc(
            batch=batch,
            num_heads=1,
            max_seq=max_seq,
            channels_head=self.kv_lora_rank,
            channels_v_head=self.channels_qk_rope_head,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(
        self,
        x: Tensor,
        *,
        positions: Tensor | None = None,
        cos_sin: tuple[Tensor, Tensor] | None = None,
        scale: float | None = None,
        is_causal: bool | None = None,
        dropout_p: float | None = None,
        attn_mask: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        out, _ = self._forward(
            x,
            positions=positions,
            cos_sin=cos_sin,
            cache=None,
            scale=scale,
            is_causal=is_causal,
            dropout_p=dropout_p,
            attn_mask=attn_mask,
            **kwargs,
        )
        return out

    def forward_cached(
        self,
        x: Tensor,
        *,
        cache: KVCache,
        positions: Tensor | None = None,
        cos_sin: tuple[Tensor, Tensor] | None = None,
        scale: float | None = None,
        is_causal: bool | None = None,
        dropout_p: float | None = None,
        attn_mask: Tensor | None = None,
        **kwargs: object,
    ) -> tuple[Tensor, KVCache]:
        """Attend using and updating the compressed latent cache.

        Args:
          x: Input tokens; [batch, seq, hidden].
          cache: KVCache to update with new compressed latents.
          positions: Token positions for RoPE; inferred from cache.seen.
          cos_sin: Precomputed RoPE (cos, sin); recomputed if None.
          scale: Attention softmax scale (Q·K^T multiplier).
          is_causal: Apply causal mask; uses config default if None.
          dropout_p: Attention dropout probability (0 at eval).
          attn_mask: Optional custom attention mask.
          **kwargs: Passed to the attention kernel.

        Returns:
          output: Attention output; [batch, seq, hidden].
          updated_cache: Updated KVCache with new seq appended.

        """
        out, updated = self._forward(
            x,
            positions=positions,
            cos_sin=cos_sin,
            cache=cache,
            scale=scale,
            is_causal=is_causal,
            dropout_p=dropout_p,
            attn_mask=attn_mask,
            **kwargs,
        )
        if updated is None:
            raise ValueError("Expected updated is not None.")
        return out, updated

    def _forward(
        self,
        x: Tensor,
        *,
        positions: Tensor | None,
        cos_sin: tuple[Tensor, Tensor] | None,
        cache: KVCache | None,
        scale: float | None,
        is_causal: bool | None,
        dropout_p: float | None,
        attn_mask: Tensor | None,
        **kwargs: object,
    ) -> tuple[Tensor, KVCache | None]:
        S = x.shape[-2]

        q = self._project_q(x)
        q_nope = q[..., : self.channels_qk_nope_head]
        q_pe = q[..., self.channels_qk_nope_head :]

        compressed = self.proj_kv_a(x)
        c_kv_new = self.kv_a_layernorm(compressed[..., : self.kv_lora_rank])
        k_pe_new = compressed[..., self.kv_lora_rank :].unsqueeze(-2)  # [*, S, 1, R].

        if cos_sin is None and self.rope is not None:
            if positions is None:
                offset = cache.seen if cache is not None else 0
                positions = torch.arange(offset, offset + S, device=x.device)
            cos_sin = self.rope(positions)
        if cos_sin is not None:
            cos, sin = cos_sin
            # DeepSeek-V3 / Kimi-K2 use the interleave pairing for the
            # decoupled RoPE slice: HF's ``apply_rotary_pos_emb`` does a
            # view→transpose→reshape pre-shuffle on q_pe/k_pe before the
            # standard rotate_half, which is equivalent to
            # ``interleave=True``. See
            # ``transformers_modules/.../modeling_deepseek.py::apply_rotary_pos_emb``.
            q_pe, k_pe_new = RoPE.rotate(q_pe, k_pe_new, cos, sin, interleave=True)

        # Cache layout: [*batch, num_heads=1, seq, feat]. ``k`` slot holds
        # c_kv, ``v`` slot holds k_pe (asymmetric dims OK).
        c_kv_cache_in = c_kv_new.unsqueeze(-3)
        k_pe_cache_in = k_pe_new.movedim(-3, -2)
        if cache is not None:
            c_kv_full_c, k_pe_full_c = cache.update(c_kv_cache_in, k_pe_cache_in)
        else:
            c_kv_full_c, k_pe_full_c = c_kv_cache_in, k_pe_cache_in

        c_kv_full = c_kv_full_c.squeeze(-3)  # [*, T, L].
        k_pe_full = k_pe_full_c.squeeze(-3)  # [*, T, R].

        causal = self.causal if is_causal is None else is_causal
        if attn_mask is None and causal:
            attn_mask = causal_chunk_mask(q_nope, k_pe_full.unsqueeze(-2))

        return self._attend(
            q_nope,
            q_pe,
            c_kv_full,
            k_pe_full,
            scale=self.softmax_scale if scale is None else scale,
            is_causal=causal,
            dropout_p=(
                (self.dropout if self.training else 0.0)
                if dropout_p is None
                else dropout_p
            ),
            attn_mask=attn_mask,
            **kwargs,
        ), cache

    # Under tensor parallelism the q-path is colwise-sharded over the head dim, so
    # ``proj_q``/``proj_q_b`` emit only this rank's ``heads_local = num_heads // tp``
    # heads as a plain local tensor; the view reshapes by ``heads_local`` (``=
    # num_heads`` when replicated).
    def _project_q(self, x: Tensor) -> Tensor:
        """Return Q as ``[..., S, heads_local, channels_qk_head]``."""
        if self.proj_q is not None:
            q = self.proj_q(x)
        else:
            if self.proj_q_a is None:
                raise ValueError("Expected self.proj_q_a is not None.")
            if self.q_a_layernorm is None:
                raise ValueError("Expected self.q_a_layernorm is not None.")
            if self.proj_q_b is None:
                raise ValueError("Expected self.proj_q_b is not None.")
            q = self.proj_q_b(self.q_a_layernorm(self.proj_q_a(x)))
        return q.view(*q.shape[:-1], self._heads_local, self.channels_qk_head)

    # The slicing lives here rather than in any kernel because it is where the tensor-
    # parallel correctness argument is: ``proj_kv_b`` stays **replicated** (the latent
    # it expands is head-shared, so sharding it over the head dim is a bug), and this
    # rank expands only its ``heads_local`` heads -- rows ``[head_offset, head_offset +
    # heads_local)`` -- so the views align with the rank-local ``q_nope``/``q_pe`` from
    # :meth:`_project_q`. A kernel receiving the views cannot get that wrong; one
    # receiving the module could.
    def _attend(
        self,
        q_nope: Tensor,
        q_pe: Tensor,
        c_kv_full: Tensor,
        k_pe_full: Tensor,
        *,
        scale: float,
        is_causal: bool,
        dropout_p: float,
        **kwargs: object,
    ) -> Tensor:
        """Slice this rank's per-head projections, then run the kernel."""
        h_local = self._heads_local
        qk_nope = self.channels_qk_nope_head

        # nn.Linear.weight is [out, in]. Out is head-major (each head's
        # (qk_nope + v) features contiguous), so view reshapes cleanly. Slice
        # this rank's local head rows before the view so the per-head index h
        # ranges over the local heads only.
        head_rows = qk_nope + self.channels_v_head
        lo = self._head_offset * head_rows
        w = self.proj_kv_b.weight[lo : lo + h_local * head_rows].view(
            h_local,
            head_rows,
            self.kv_lora_rank,
        )
        out_per_head = self.attn_kernel(
            q_nope,
            q_pe,
            c_kv_full,
            k_pe_full,
            w_kr=w[:, :qk_nope, :],
            w_uv=w[:, qk_nope:, :],
            scale=scale,
            is_causal=is_causal,
            dropout_p=dropout_p,
            **kwargs,
        )
        return self.proj_out(self._to_proj_out_input(out_per_head.flatten(-2)))

    # Replicated: a plain ``[..., num_heads * v]`` tensor. Under tensor parallelism
    # ``proj_out`` is rowwise-sharded and expects its input sharded on the last (head*v)
    # dim, so wrap this rank's local ``[..., heads_local * v]`` slice as a ``Shard(-1)``
    # DTensor; ``RowwiseParallel`` then all-reduces the partial outputs into the
    # replicated result.
    def _to_proj_out_input(self, out: Tensor) -> Tensor:
        """Present the per-head output to ``proj_out`` in its expected layout."""
        if self._tp_mesh is None:
            return out
        return DTensor.from_local(out, self._tp_mesh, [Shard(-1)], run_check=False)

    def tensor_parallel_style(self) -> ParallelStyle:
        """Return the head-parallel ``ParallelStyle`` for this MLA block.

        MLA is the hard tier of the two-tier tensor-parallel contract: its
        absorb-math reshapes ``proj_kv_b.weight`` per head in Python, which the
        generic colwise/rowwise styles cannot express. The style shards the
        q-path (``proj_q``/``proj_q_b``, colwise over the head dim) and
        ``proj_out`` (rowwise), records this rank's local head range so the
        forward expands only its heads, and leaves the head-shared latent path
        (``proj_kv_a``/``proj_kv_b``/``proj_q_a``/``kv_a_layernorm``)
        replicated -- sharding the latent over the head axis is a correctness
        bug, not merely wasteful.

        Returns:
          style: Custom _MLAParallel instance for head-dim sharding.

        Raises:
          ValueError: If ``tp`` does not divide ``num_heads`` (the per-rank head
            range would be ragged).

        """
        return _MLAParallel()

    def shard_heads_over(self, mesh: DeviceMesh) -> None:
        """Record this rank's local head range for head-parallel forward.

        Validates that ``mesh`` (the ``tp`` sub-mesh) divides ``num_heads``, then
        stores ``heads_local`` and ``head_offset`` so :meth:`_project_q` and
        :meth:`_attend` expand only this rank's num_heads and
        :meth:`_to_proj_out_input` wraps the per-head output over ``mesh``.

        Args:
          mesh: The ``tp`` device sub-mesh the head dim is sharded over.

        Raises:
          ValueError: If ``mesh.size()`` does not divide ``num_heads``, or the
            kernel delegates to one with no DTensor sharding strategy.

        """
        self.assert_shardable_over(mesh.size())
        self._tp_mesh = mesh
        self._heads_local = self.num_heads // mesh.size()
        self._head_offset = mesh.get_local_rank() * self._heads_local

    def assert_shardable_over(self, tp_size: int) -> None:
        """Reject a sharding this block's configuration cannot express.

        Deliberately NOT named ``assert_tensor_parallel_compatible``: that is
        :class:`~priml.train.tensor_parallel.TensorParallelValidator`'s
        zero-argument method, which ``apply_tensor_parallel`` calls on every
        submodule AFTER sharding (``train/tensor_parallel.py:140``). Adding a
        required parameter to a method of that name silently leaves the
        protocol, and the call site then fails with a ``TypeError`` about a
        missing argument instead of the check running.

        Split from :meth:`shard_heads_over` because it reads only
        CONFIGURATION -- head count and injected kernel -- so it answers
        without an initialized process group.

        Args:
          tp_size: Ranks the head dim would be split across.

        Raises:
          ValueError: ``tp_size`` does not divide ``num_heads``, or the kernel
            delegates to one with no DTensor sharding strategy.

        """
        if self.num_heads % tp_size != 0:
            raise ValueError(
                f"Tensor-parallel size {tp_size} must divide num_heads "
                f"{self.num_heads} (MLA shards the head dim; an indivisible "
                f"count yields a ragged per-rank head range).",
            )
        # Same gap ``SelfAttention.assert_tensor_parallel_compatible`` covers:
        # the fused flash kernel has no DTensor sharding strategy and dies deep
        # in the dispatcher rather than here, naming a stride.
        if isinstance(self.attn_kernel, LatentAttention) and isinstance(
            self.attn_kernel.attn_kernel,
            SdpaFused,
        ):
            raise ValueError(  # noqa: TRY004 -- The configuration is invalid for this kernel, not invalid because of a caller type.
                "Tensor parallelism requires a DTensor-compatible attention "
                "kernel; set the latent kernel's attn_kernel to SdpaNaive "
                "(the fused flash kernel has no DTensor sharding strategy).",
            )

    def tensor_parallel_plan(self) -> dict[str, ParallelStyle]:
        """Which of THIS module's children shard, and how.

        ``parallelize_module`` addresses children by attribute name, so some
        object has to supply those names -- and it must be the module that owns
        them. Answering here rather than from the style keeps the projections
        injectable: a caller may fill ``proj_q`` with any module, and the plan
        still finds it, because the plan names the ATTRIBUTE this class binds
        rather than the class the caller chose.

        Only the per-head path shards. ``proj_q`` XOR ``proj_q_b`` exists (the
        ``q_lora_rank`` gate); whichever emits head-major q is colwise, with a
        LOCAL output so the absorb-math einsums see plain tensors rather than
        DTensors. ``proj_out`` is rowwise. Everything else -- the latent path and
        its norms -- stays replicated, which is correctness rather than thrift:
        the latent is head-shared, so splitting it over the head dim is wrong.

        Returns:
          plan: Attribute name to style, for ``parallelize_module``.

        """
        q_name = "proj_q" if self.proj_q is not None else "proj_q_b"
        return {
            q_name: ColwiseParallel(use_local_output=True),
            "proj_out": RowwiseParallel(
                input_layouts=Shard(-1),
                output_layouts=Replicate(),
            ),
        }


# A field left at its declared default inherits the parent's value; anything else the
# caller chose outranks it. A field set to exactly the default is indistinguishable from
# an untouched one.
def _fill_unset(config: Linear.Config, name: str, value: object) -> None:
    """Push ``value`` onto ``config.name`` unless the caller set it."""
    # Read off the dataclass field: slots=True makes the class attribute a
    # descriptor rather than the default value.
    default = next(f for f in fields(config) if f.name == name).default
    if getattr(config, name) == default:
        setattr(config, name, value)


class _MLAParallel(ParallelStyle):
    """Head-parallel tensor-parallel style for :class:`MultiHeadLatentAttention`.

    Holds no knowledge of MLA's children: it records this rank's head range and
    applies the plan the module itself supplies, so adding or renaming a
    projection is a change to one class rather than to two.
    """

    @override
    def _apply(
        self,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> nn.Module:
        assert isinstance(module, MultiHeadLatentAttention)
        module.shard_heads_over(device_mesh)
        return parallelize_module(module, device_mesh, module.tensor_parallel_plan())
