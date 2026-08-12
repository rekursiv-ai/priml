"""Multi-head Latent Attention (MLA).

Used by DeepSeek-V2/V3 and Kimi-K2. Compresses KV through a low-rank
``kv_lora_rank`` latent, applies RoPE only on a decoupled
``qk_rope_head_dim`` slice, and uses a separate ``v_head_dim`` for the
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

Output: ``[B, S, n, v_head_dim]`` → ``o_proj`` → ``[B, S, hidden]``.

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

from dataclasses import KW_ONLY
from typing import TYPE_CHECKING, Any, Literal, Self, override

from configgle import Fig
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

        qk_nope_head_dim: int = 128
        """Non-RoPE portion of the Q/K head dim."""

        qk_rope_head_dim: int = 64
        """RoPE portion of the Q/K head dim (shared key for all heads)."""

        v_head_dim: int = 128
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

        rope: RoPE.Config | None = None
        """RoPE config applied to ``qk_rope`` slice. ``None`` ⇒ no RoPE."""

        rms_norm_eps: float = 1e-6
        """Epsilon for the LoRA RMSNorm layers."""

        init_weight: InitFn = kaiming_uniform
        """Weight init for linear projections."""

        depth: int = -1
        """Block depth for depth-scaled init (-1 = no scaling)."""

        shard: Literal["none", "colwise"] = "none"
        """Tensor-parallel shard style over the mesh tp dim; none = replicated.

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
        def qk_head_dim(self) -> int:
            return self.qk_nope_head_dim + self.qk_rope_head_dim

        @override
        def finalize(self) -> Self:
            if self.channels_in < 1:
                raise ValueError(
                    f"channels_in must be > 0, got {self.channels_in}.",
                )
            if self.heads < 1:
                raise ValueError(f"heads must be > 0, got {self.heads}.")
            if self.kv_lora_rank < 1:
                raise ValueError(
                    f"kv_lora_rank must be > 0, got {self.kv_lora_rank}.",
                )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.heads = config.heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.dropout = config.dropout
        self.causal = config.causal
        self.depth = config.depth
        self.softmax_scale = config.softmax_scale or self.qk_head_dim**-0.5
        self.shard = config.shard

        c = config.channels_in

        # Tensor parallelism is head-parallel: the custom ``ParallelStyle``
        # (see ``tensor_parallel_style``) shards the q-path and ``o_proj`` over
        # the head dim and makes the absorb-math rank-local. Until that style is
        # applied these record the replicated identity (every rank owns all
        # heads); the head-shared latent path always stays replicated.
        self._tp_mesh: DeviceMesh | None = None
        self._heads_local = config.heads
        self._head_offset = 0

        # Q path
        if config.q_lora_rank is None:
            self.q_proj = Linear.Config(
                channels_in=c,
                channels_out=config.heads * self.qk_head_dim,
                bias=config.bias,
                depth=config.depth,
                init_weight=config.init_weight,
            ).make()
            self.q_a_proj: nn.Module | None = None
            self.q_a_layernorm: nn.Module | None = None
            self.q_b_proj: nn.Module | None = None
        else:
            self.q_proj = None
            self.q_a_proj = Linear.Config(
                channels_in=c,
                channels_out=config.q_lora_rank,
                bias=config.bias,
                depth=config.depth,
                init_weight=config.init_weight,
            ).make()
            self.q_a_layernorm = RMSNorm.Config(
                channels_in=config.q_lora_rank,
                eps=config.rms_norm_eps,
                elementwise_affine=True,
            ).make()
            self.q_b_proj = Linear.Config(
                channels_in=config.q_lora_rank,
                channels_out=config.heads * self.qk_head_dim,
                bias=config.bias,
                depth=config.depth,
                init_weight=config.init_weight,
            ).make()

        # KV path: compressed_kv is split into (c_kv, k_pe).
        self.kv_a_proj = Linear.Config(
            channels_in=c,
            channels_out=config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.bias,
            depth=config.depth,
            init_weight=config.init_weight,
        ).make()
        self.kv_a_layernorm = RMSNorm.Config(
            channels_in=config.kv_lora_rank,
            eps=config.rms_norm_eps,
            elementwise_affine=True,
        ).make()
        # kv_b expands c_kv into (n_heads * (qk_nope + v_head_dim)).
        self.kv_b_proj = Linear.Config(
            channels_in=config.kv_lora_rank,
            channels_out=config.heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=config.bias,
            depth=config.depth,
            init_weight=config.init_weight,
        ).make()
        self.o_proj = Linear.Config(
            channels_in=config.heads * config.v_head_dim,
            channels_out=c,
            bias=config.bias,
            depth=config.depth,
            init_weight=config.init_weight,
        ).make()

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
            channels_head_v=self.qk_rope_head_dim,
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
        q_nope = q[..., : self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim :]

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
        """Return Q as ``[..., S, heads_local, qk_head_dim]``.

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
        return q.view(*q.shape[:-1], self._heads_local, self.qk_head_dim)

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
        qk_nope = self.qk_nope_head_dim
        v_dim = self.v_head_dim

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


class _MLAParallel(ParallelStyle):
    """Head-parallel tensor-parallel style for :class:`MultiHeadLatentAttention`.

    Shards the q-path (``q_proj``/``q_b_proj``, colwise over the head dim) and
    ``o_proj`` (rowwise), keeping each rank's q output a plain local tensor so
    the absorb-math einsums run on plain operands (no DTensor mixed-operand
    failure). The head-shared latent path stays replicated. The style records
    this rank's local head range on the module so :meth:`_project_q` and
    :meth:`_absorb_attention` expand only its ``heads // tp`` heads.
    """

    @override
    def _apply(
        self,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> nn.Module:
        assert isinstance(module, MultiHeadLatentAttention)
        module.shard_heads_over(device_mesh)

        # q_proj XOR q_b_proj is present (q_lora_rank gate); shard whichever
        # emits the per-head q. Both are colwise over the head-major output, and
        # the local output stays a plain tensor for the rank-local absorb-math.
        q_name = "q_proj" if module.q_proj is not None else "q_b_proj"
        plan: dict[str, ParallelStyle] = {
            q_name: ColwiseParallel(use_local_output=True),
            "o_proj": RowwiseParallel(
                input_layouts=Shard(-1),
                output_layouts=Replicate(),
            ),
        }
        return parallelize_module(module, device_mesh, plan)
