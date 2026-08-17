"""Multi-head Latent Attention (MLA).

Used by DeepSeek-V2/V3 and Kimi-K2. Compresses KV through a low-rank
``kv_lora_rank`` latent, applies RoPE only on a decoupled
``channels_qk_rope_head`` slice, and uses a separate ``channels_v_head`` for the
value path.

Q has an optional LoRA decomposition (``q_lora_rank``): DeepSeek-V3 uses
it (1536), Kimi-K2 does not (None ⇒ single ``q_proj``).

Forward shape map (B=batch, S=seq, n=num_heads)::

    x [B, S, hidden]
      ├─ q_path  -> q  [B, S, n, qk_nope + qk_rope]
      │            ├─ q_nope [B, S, n, qk_nope]
      │            └─ q_pe   [B, S, n, qk_rope]   -> RoPE
      └─ kv_path -> c_kv [B, S, kv_lora_rank]     (post-kv_a_layernorm)
                   k_pe  [B, S, qk_rope]          (post-RoPE, head-shared)

    At attention time, kv_b_proj expands c_kv -> (k_nope, v) per head.

Output: ``[B, S, n, channels_v_head]`` → ``o_proj`` → ``[B, S, hidden]``.

**Cache layout.** Only ``(c_kv, k_pe)`` are cached — not the expanded
K/V. For Kimi-K2 (n=64, qk_nope+v=256, kv_lora_rank=512, qk_rope=64)
this cuts cache memory ~25× (576 dims/token vs 16384).

**Absorb-math forward.** Default path reshapes ``kv_b_proj.weight``
into per-head ``W_KR`` (qk_nope slice) and ``W_UV`` (v_head slice),
then fuses Q projection with the K expansion::

    q_abs[b,s,h,l] = sum_d q_nope[b,s,h,d] * W_KR[h,d,l]
    attn_logits    = q_abs · c_kv^T + q_pe · k_pe^T
    out_latent     = attn · c_kv
    out[b,s,h,v]   = sum_l out_latent[b,h,s,l] * W_UV[h,v,l]

Decode-step compute: O(T·H·L) rather than O(T·H·(qk_nope+v)·L) for
re-expand; the ``(qk_nope+v)/L`` factor (~0.5 on Kimi-K2) is saved on
every step. Numerical output is identical up to softmax ordering;
verified against a naive re-expand reference in the test suite.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Self, override

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    parallelize_module,
)
from torch.nn import functional as f

import torch

from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    RotaryFactors,
    TensorModule,
)
from priml.model.init import InitFn, kaiming_uniform
from priml.model.kvcache import KVCache
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.rope import RoPE


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


class MultiHeadLatentAttention(nn.Module):
    """MLA with decoupled RoPE and compressed KV latent."""

    class Config(Fig["MultiHeadLatentAttention"], kw_only=False):
        channels_in: int = -1
        """Model width."""

        _: KW_ONLY

        heads: int = -1
        """Number of attention heads (shared by Q, K, V)."""

        channels_qk_nope_head: int = 128
        """Non-RoPE portion of the Q/K head dim."""

        channels_qk_rope_head: int = 64
        """RoPE portion of the Q/K head dim (shared key for all heads)."""

        channels_v_head: int = 128
        """Value head dim (independent of Q/K head dim)."""

        q_lora_rank: int | None = None
        """Rank for Q LoRA. ``None`` ⇒ single ``q_proj`` (Kimi-K2). Non-None
        uses ``q_a_proj`` + RMSNorm + ``q_b_proj`` (DeepSeek-V3)."""

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

        rope: Makeable[RotaryFactors] | None = None
        """Rotary embedding applied to the ``qk_rope`` slice. ``None`` ⇒ none.

        Typed by what MLA CALLS -- positions in, ``(cos, sin)`` out -- rather
        than by ``RoPE``, so a learned or NTK-scaled variant fills the slot."""

        norm_q_lora: Makeable[nn.Module] = field(
            default_factory=partial(RMSNorm.Config, elementwise_affine=True),
        )
        """Normalization between the Q LoRA's two projections.

        Built only when ``q_lora_rank`` is set; its width is the rank."""

        proj_q: Makeable[nn.Module] = field(default_factory=Linear.Config)
        """Q projection: model width in, ``heads * channels_qk_head`` out.

        Built only when ``q_lora_rank`` is None; the LoRA path uses
        ``proj_q_a``/``proj_q_b`` instead. Sharded head-parallel under tensor
        parallelism (see :meth:`tensor_parallel_plan`)."""

        proj_q_a: Makeable[nn.Module] = field(default_factory=Linear.Config)
        """Q LoRA down-projection to ``q_lora_rank``. Replicated: the rank is
        head-shared, so sharding it over the head dim is a correctness bug."""

        proj_q_b: Makeable[nn.Module] = field(default_factory=Linear.Config)
        """Q LoRA up-projection to ``heads * channels_qk_head``, head-major.
        Sharded head-parallel, like ``proj_q``."""

        proj_kv_a: Makeable[nn.Module] = field(default_factory=Linear.Config)
        """KV down-projection to ``kv_lora_rank + channels_qk_rope_head``.
        Replicated -- this is the latent the cache holds."""

        proj_kv_b: Makeable[nn.Module] = field(default_factory=Linear.Config)
        """KV latent expansion. Replicated even under tensor parallelism: the
        absorb math slices its weight per head in Python, so each rank holds the
        whole thing and expands only its own head range."""

        proj_out: Makeable[TensorModule] = field(default_factory=Linear.Config)
        """Output projection back to model width; rowwise-sharded.

        Typed by its CALL shape, unlike the others: this is the one projection
        whose result is returned directly, so a bare ``nn.Module`` (whose
        ``__call__`` is untyped) would make the forward's return ``Any``."""

        norm_kv_lora: Makeable[nn.Module] = field(
            default_factory=partial(RMSNorm.Config, elementwise_affine=True),
        )
        """Normalization on the compressed KV latent; its width is
        ``kv_lora_rank``."""

        init_weight: InitFn = kaiming_uniform
        """Weight init for linear projections."""

        depth: int = -1
        """Block depth for depth-scaled init (-1 = no scaling)."""

        shard: Literal["colwise"] | None = None
        """Tensor-parallel shard style over the mesh tp dim; ``None`` replicates.

        ``"colwise"`` selects the head-parallel custom style (see
        :meth:`MultiHeadLatentAttention.tensor_parallel_style`): the q-path and
        ``o_proj`` shard over the head dim while the head-shared latent path
        stays replicated. MLA exposes no ``"rowwise"`` -- it is sharded as one
        block, not per child projection.
        """

        @property
        def channels_out(self) -> int:
            return self.channels_in

        @property
        def channels_qk_head(self) -> int:
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

        def _size_projections(self) -> None:
            """Fill each projection slot's widths from the shape fields.

            Every width here is DERIVED -- a head count times a per-head width,
            or a LoRA rank -- so a caller states the shape once and swaps the
            projection class without restating any of it. A slot that already
            carries a width (a caller sized it deliberately) is left alone.
            """
            qk_out = self.heads * self.channels_qk_head
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
                    self.heads * (self.channels_qk_nope_head + self.channels_v_head),
                ),
                (self.proj_out, self.heads * self.channels_v_head, self.channels_in),
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
                    slot.bias = self.bias
                    slot.depth = self.depth
                    slot.init_weight = self.init_weight

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.channels_in < 1:
            raise ValueError(f"channels_in must be > 0, got {config.channels_in}.")
        if config.heads < 1:
            raise ValueError(f"heads must be > 0, got {config.heads}.")
        if config.kv_lora_rank < 1:
            raise ValueError(f"kv_lora_rank must be > 0, got {config.kv_lora_rank}.")
        self.heads = config.heads
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
        self.depth = config.depth
        self.softmax_scale = config.softmax_scale or self.channels_qk_head**-0.5
        self.shard = config.shard

        # Tensor parallelism is head-parallel: the custom ``ParallelStyle``
        # (see ``tensor_parallel_style``) shards the q-path and ``o_proj`` over
        # the head dim and makes the absorb-math rank-local. Until that style is
        # applied these record the replicated identity (every rank owns all
        # heads); the head-shared latent path always stays replicated.
        self._tp_mesh: DeviceMesh | None = None
        self._heads_local = config.heads
        self._head_offset = 0

        # Q path. The attribute names below are the CONTRACT the tensor-parallel
        # plan names (see ``tensor_parallel_plan``); the slots they are built
        # from are the caller's choice.
        if config.q_lora_rank is None:
            self.q_proj = config.proj_q.make()
            self.q_a_proj: nn.Module | None = None
            self.q_a_layernorm: nn.Module | None = None
            self.q_b_proj: nn.Module | None = None
        else:
            self.q_proj = None
            self.q_a_proj = config.proj_q_a.make()
            self.q_a_layernorm = config.norm_q_lora.make()
            self.q_b_proj = config.proj_q_b.make()

        # KV path: compressed_kv is split into (c_kv, k_pe).
        self.kv_a_proj = config.proj_kv_a.make()
        self.kv_a_layernorm = config.norm_kv_lora.make()
        # kv_b expands c_kv into (n_heads * (qk_nope + channels_v_head)).
        self.kv_b_proj = config.proj_kv_b.make()
        self.o_proj = config.proj_out.make()

        self.rope = config.rope.make() if config.rope else None

    def reset_parameters(self) -> None:
        for m in (
            self.q_proj,
            self.q_a_proj,
            self.q_a_layernorm,
            self.q_b_proj,
            self.kv_a_proj,
            self.kv_a_layernorm,
            self.kv_b_proj,
            self.o_proj,
        ):
            if m is not None and hasattr(m, "reset_parameters"):
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
        """
        return KVCache.alloc(
            batch=batch,
            heads=1,
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
        *args: Any,
        positions: Tensor | None = None,
        cos_sin: tuple[Tensor, Tensor] | None = None,
        cache: KVCache | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, KVCache]:
        del args, kwargs
        S = x.shape[-2]

        q = self._project_q(x)
        q_nope = q[..., : self.channels_qk_nope_head]
        q_pe = q[..., self.channels_qk_nope_head :]

        compressed = self.kv_a_proj(x)
        c_kv_new = self.kv_a_layernorm(compressed[..., : self.kv_lora_rank])
        k_pe_new = compressed[..., self.kv_lora_rank :].unsqueeze(-2)  # [*, S, 1, R]

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

        # Cache layout: [*batch, heads=1, seq, feat]. ``k`` slot holds
        # c_kv, ``v`` slot holds k_pe (asymmetric dims OK).
        c_kv_cache_in = c_kv_new.unsqueeze(-3)
        k_pe_cache_in = k_pe_new.movedim(-3, -2)
        if cache is not None:
            c_kv_full_c, k_pe_full_c = cache.update(c_kv_cache_in, k_pe_cache_in)
        else:
            cache = KVCache(c_kv_cache_in, k_pe_cache_in)
            c_kv_full_c, k_pe_full_c = c_kv_cache_in, k_pe_cache_in

        T = c_kv_full_c.shape[-2]
        c_kv_full = c_kv_full_c.squeeze(-3)  # [*, T, L]
        k_pe_full = k_pe_full_c.squeeze(-3)  # [*, T, R]

        out = self._absorb_attention(q_nope, q_pe, c_kv_full, k_pe_full, S, total_len=T)
        return out, cache

    def _project_q(self, x: Tensor) -> Tensor:
        """Return Q as ``[..., S, heads_local, channels_qk_head]``.

        Under tensor parallelism the q-path is colwise-sharded over the head
        dim, so ``q_proj``/``q_b_proj`` emit only this rank's
        ``heads_local = heads // tp`` heads as a plain local tensor; the view
        reshapes by ``heads_local`` (``= heads`` when replicated).
        """
        if self.q_proj is not None:
            q = self.q_proj(x)
        else:
            assert self.q_a_proj is not None
            assert self.q_a_layernorm is not None
            assert self.q_b_proj is not None
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        assert isinstance(q, Tensor)
        return q.view(*q.shape[:-1], self._heads_local, self.channels_qk_head)

    def _absorb_attention(
        self,
        q_nope: Tensor,
        q_pe: Tensor,
        c_kv_full: Tensor,
        k_pe_full: Tensor,
        seq_len: int,
        *,
        total_len: int,
    ) -> Tensor:
        """Absorb-math MLA attention (rank-local under tensor parallelism).

        Reshapes ``kv_b_proj.weight`` into per-head ``W_KR`` (qk_nope
        slice) and ``W_UV`` (v slice) via a view — no copy — and
        attends directly in the latent space. Output equivalent to
        the re-expand form up to softmax reordering.

        ``kv_b_proj`` stays **replicated** (the latent it expands is
        head-shared, so sharding it is a correctness bug); under tensor
        parallelism this rank holds the full weight but expands only its
        ``heads_local`` heads — the rows ``[head_offset, head_offset +
        heads_local)`` — so ``W_KR``/``W_UV`` align with the rank-local
        ``q_nope``/``q_pe`` from :meth:`_project_q`.
        """
        h_local = self._heads_local
        L = self.kv_lora_rank
        qk_nope = self.channels_qk_nope_head
        v_dim = self.channels_v_head

        # nn.Linear.weight is [out, in]. Out is head-major (each head's
        # (qk_nope + v) features contiguous), so view reshapes cleanly. Slice
        # this rank's local head rows before the view so the per-head index h
        # ranges over the local heads only.
        head_rows = qk_nope + v_dim
        lo = self._head_offset * head_rows
        w = self.kv_b_proj.weight[lo : lo + h_local * head_rows].view(
            h_local, head_rows, L
        )
        w_kr = w[:, :qk_nope, :]
        w_uv = w[:, qk_nope:, :]

        # q_abs[..., s, h, l] = sum_d q_nope[..., s, h, d] * W_KR[h, d, l].
        q_abs = torch.einsum("...shd,hdl->...shl", q_nope, w_kr)

        # Attention logits: latent nope + rope term. k_pe is head-
        # shared so contract directly against the shared tensor.
        attn_nope = torch.einsum("...shl,...tl->...hst", q_abs, c_kv_full)
        attn_rope = torch.einsum("...shr,...tr->...hst", q_pe, k_pe_full)
        logits = (attn_nope + attn_rope) * self.softmax_scale

        if self.causal and seq_len > 1:
            # Query ``i`` sits at absolute position ``total_len - seq_len
            # + i`` and may attend to keys ``0..that``; mask the strictly
            # later keys. Bottom-right aligned for cached multi-token chunks.
            mask = torch.full(
                (seq_len, total_len),
                float("-inf"),
                device=logits.device,
                dtype=logits.dtype,
            ).triu(total_len - seq_len + 1)
            logits = logits + mask

        attn = logits.softmax(dim=-1)
        if self.training and self.dropout > 0:
            attn = f.dropout(attn, p=self.dropout)

        # Output in latent space, then project to v via W_UV.
        out_latent = torch.einsum("...hst,...tl->...hsl", attn, c_kv_full)
        out_per_head = torch.einsum("...hsl,hvl->...shv", out_latent, w_uv)
        return self.o_proj(self._to_o_proj_input(out_per_head.flatten(-2)))

    def _to_o_proj_input(self, out: Tensor) -> Tensor:
        """Present the per-head output to ``o_proj`` in its expected layout.

        Replicated: a plain ``[..., heads * v]`` tensor. Under tensor
        parallelism ``o_proj`` is rowwise-sharded and expects its input sharded
        on the last (head*v) dim, so wrap this rank's local ``[..., heads_local
        * v]`` slice as a ``Shard(-1)`` DTensor; ``RowwiseParallel`` then
        all-reduces the partial outputs into the replicated result.
        """
        if self._tp_mesh is None:
            return out
        return DTensor.from_local(out, self._tp_mesh, [Shard(-1)], run_check=False)

    def tensor_parallel_style(self) -> ParallelStyle:
        """Return the head-parallel ``ParallelStyle`` for this MLA block.

        MLA is the hard tier of the two-tier tensor-parallel contract: its
        absorb-math reshapes ``kv_b_proj.weight`` per head in Python, which the
        generic colwise/rowwise styles cannot express. The style shards the
        q-path (``q_proj``/``q_b_proj``, colwise over the head dim) and
        ``o_proj`` (rowwise), records this rank's local head range so the
        forward expands only its heads, and leaves the head-shared latent path
        (``kv_a_proj``/``kv_b_proj``/``q_a_proj``/``kv_a_layernorm``)
        replicated -- sharding the latent over the head axis is a correctness
        bug, not merely wasteful.

        Raises:
          ValueError: If ``tp`` does not divide ``heads`` (the per-rank head
            range would be ragged).

        """
        return _MLAParallel()

    def shard_heads_over(self, mesh: DeviceMesh) -> None:
        """Record this rank's local head range for head-parallel forward.

        Validates that ``mesh`` (the ``tp`` sub-mesh) divides ``heads``, then
        stores ``heads_local`` and ``head_offset`` so :meth:`_project_q` and
        :meth:`_absorb_attention` expand only this rank's heads and
        :meth:`_to_o_proj_input` wraps the per-head output over ``mesh``.

        Args:
          mesh: The ``tp`` device sub-mesh the head dim is sharded over.

        Raises:
          ValueError: If ``mesh.size()`` does not divide ``heads``.

        """
        tp_size = mesh.size()
        if self.heads % tp_size != 0:
            raise ValueError(
                f"Tensor-parallel size {tp_size} must divide heads "
                f"{self.heads} (MLA shards the head dim; an indivisible "
                f"count yields a ragged per-rank head range).",
            )
        self._tp_mesh = mesh
        self._heads_local = self.heads // tp_size
        self._head_offset = mesh.get_local_rank() * self._heads_local

    def tensor_parallel_plan(self) -> dict[str, ParallelStyle]:
        """Which of THIS module's children shard, and how.

        ``parallelize_module`` addresses children by attribute name, so some
        object has to supply those names -- and it must be the module that owns
        them. Answering here rather than from the style keeps the projections
        injectable: a caller may fill ``proj_q`` with any module, and the plan
        still finds it, because the plan names the ATTRIBUTE this class binds
        rather than the class the caller chose.

        Only the per-head path shards. ``q_proj`` XOR ``q_b_proj`` exists (the
        ``q_lora_rank`` gate); whichever emits head-major q is colwise, with a
        LOCAL output so the absorb-math einsums see plain tensors rather than
        DTensors. ``o_proj`` is rowwise. Everything else -- the latent path and
        its norms -- stays replicated, which is correctness rather than thrift:
        the latent is head-shared, so splitting it over the head dim is wrong.

        Returns:
          plan: Attribute name to style, for ``parallelize_module``.

        """
        q_name = "q_proj" if self.q_proj is not None else "q_b_proj"
        return {
            q_name: ColwiseParallel(use_local_output=True),
            "o_proj": RowwiseParallel(
                input_layouts=Shard(-1),
                output_layouts=Replicate(),
            ),
        }


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
