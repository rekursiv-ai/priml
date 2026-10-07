"""Attention kernel implementations."""

from __future__ import annotations

from typing import override

import math

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import (
    Cost,
    elementwise_cost,
    matmul_cost,
    reduction_cost,
    traffic,
)
from priml.model.attention.window import (
    causal_chunk_mask,
    combined_mask,
    segment_mask,
)


def attention_kernel_cost(
    *,
    seq_len: int,
    batch_size: int = 1,
    dtype: torch.dtype | None,
    num_heads: int,
    channels_head: int,
    channels_v_head: int = -1,
    num_heads_kv: int = -1,
    window: int = -1,
    dropout_p: float = 0.0,
    rows: int = -1,
    **kwargs: object,
) -> Cost:
    """Cost the complete batched ``softmax(QK^T)V`` invocation.

    A kernel config holds no shapes, so its ``cost`` takes them from the OWNER
    of the projections -- how many heads and how wide -- the way ``forward``
    takes ``q``/``k``/``v``. Every kernel whose products are the standard two
    binds this as its ``cost``. Counted over the full window with no causal
    discount, per the module policy.

    Args:
      seq_len: Tokens per sequence: the keys a query reaches before any window.
      batch_size: Sequences in this invocation.
      dtype: Activation dtype; ``None`` is torch's default.
      num_heads: Query heads.
      num_heads_kv: Key/value heads; ``-1`` mirrors ``num_heads``. A smaller
        count is Grouped-Query Attention, where several query heads share one
        key and value head.
      channels_head: Width of each query/key head.
      channels_v_head: Value width; -1 uses the query/key width.
      window: Previous keys each query reaches, plus itself; negative is unbounded.
      dropout_p: Attention dropout rate; nonzero adds a mask and a rescale.
      rows: Query rows sharing K/V, or ``-1`` to use the key count.
      **kwargs: The rest of the owner's bus, unread.

    Returns:
      cost: Unfused logical tensor I/O and FLOPs, not measured HBM traffic.
        Fused and naive kernels share this algorithmic accounting convention.

    Raises:
      ValueError: A positive ``num_heads_kv`` exceeds ``num_heads`` or does not
        divide it, so no whole group of query heads shares one key/value head.

    """
    del kwargs
    _validate_kv_heads(num_heads=num_heads, num_heads_kv=num_heads_kv)
    dt = dtype
    keys = seq_len if window < 0 else min(window + 1, seq_len)
    query_rows = (seq_len if rows < 0 else rows) * batch_size
    value_width = channels_head if channels_v_head < 0 else channels_v_head
    # Each batch carries its own K/V tensors; tile whole per-sequence products.
    sequence_rows = seq_len if rows < 0 else rows
    scores = matmul_cost(
        channels_in=channels_head,
        channels_out=keys,
        weight=False,
        rows=sequence_rows,
        dtype=dt,
        operand_read=False,
    ).tile(batch_size)
    values = matmul_cost(
        channels_in=keys,
        channels_out=value_width,
        weight=False,
        rows=sequence_rows,
        dtype=dt,
        operand_read=False,
    ).tile(batch_size)
    # Logical unfused I/O, including scores, even when execution uses fused SDPA.
    # Scale/subtract/exp/divide read two row scalars; VJP reads one row sum.
    softmax = (
        traffic(
            "primal",
            "elementwise",
            elements=8 * keys + 2,
            flops=4 * keys,
            dtype=dt,
        )
        + reduction_cost(input_elements=keys, dtype=dt).tile(2, copies=2)
        + traffic(
            "adjoint",
            "elementwise",
            elements=10 * keys + 1,
            flops=4 * keys,
            dtype=dt,
        )
        + reduction_cost(input_elements=keys, dtype=dt, phase="adjoint")
    ).tile(query_rows)
    dropout = elementwise_cost(
        primal=2 * keys * query_rows if dropout_p > 0 else 0,
        adjoint=2 * keys * query_rows if dropout_p > 0 else 0,
        channels=keys if dropout_p > 0 else 0,
        inputs=3,
        outputs=2,
        adjoint_inputs=2,
        rows=query_rows,
        dtype=dt,
    )
    # Grouped-query attention shares K and V across the query heads in a group,
    # so those operand reads happen once per KV head rather than once per query
    # head -- charging them per query head overstates them by num_heads /
    # num_heads_kv. The two products above therefore leave the operand read out
    # and it is added back here, at the KV head count. The arithmetic stays per
    # query head: each one really does run its own two products over the whole
    # context, so FLOPs never move.
    #
    # The read those products left out, rebuilt as plain operand traffic: K of
    # ``channels_head x keys`` and V of ``keys x value_width`` per KV head, per
    # sequence. The adjoint moves twice the primal, matching ``matmul_cost``'s
    # own adjoint cell.
    kv_elements = channels_head * keys + keys * value_width
    kv_read = (
        traffic("primal", "matmul", elements=kv_elements, dtype=dt)
        + traffic("adjoint", "matmul", elements=2 * kv_elements, dtype=dt)
    ).tile(batch_size)
    per_head = (scores + values + softmax + dropout).tile(num_heads, copies=num_heads)
    return per_head + kv_read.tile(_kv_heads(num_heads, num_heads_kv))


def _kv_heads(num_heads: int, num_heads_kv: int) -> int:
    """Key/value heads to charge the shared K/V read at; ``-1`` mirrors queries."""
    return num_heads if num_heads_kv <= 0 else num_heads_kv


def _validate_kv_heads(*, num_heads: int, num_heads_kv: int) -> None:
    """Reject a KV head count that no whole group of query heads can produce."""
    if num_heads_kv <= 0:
        return
    if num_heads_kv > num_heads or num_heads % num_heads_kv != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by num_heads_kv={num_heads_kv}; "
            "each group of query heads shares exactly one key/value head.",
        )


class SdpaFused(nn.Module):
    """Wraps F.scaled_dot_product_attention.

    Takes ``[..., S, num_heads, channels_head]`` -- the layout every priml
    projection emits and the one a fused kernel wants -- and transposes to
    SDPA's ``[..., num_heads, S, channels_head]`` internally. The transpose is a
    stride view rather than a copy, so a kernel that needs the other layout
    (FlashAttention-3) is a drop-in value in the same slot and pays nothing.
    """

    class Config(Fig["SdpaFused"]):
        @classmethod
        def cost(
            cls,
            *,
            seq_len: int,
            batch_size: int = 1,
            dtype: torch.dtype | None,
            num_heads: int,
            channels_head: int,
            channels_v_head: int = -1,
            num_heads_kv: int = -1,
            window: int = -1,
            dropout_p: float = 0.0,
            rows: int = -1,
            **kwargs: object,
        ) -> Cost:
            """Cost the kernel from the shapes its owner hands it.

            See :func:`attention_kernel_cost` for every argument.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              num_heads: Query heads.
              num_heads_kv: Key/value heads; -1 mirrors num_heads.
              channels_head: Width of each query/key head.
              channels_v_head: Value width; -1 uses the query/key width.
              window: Previous keys each query reaches, plus itself; negative is unbounded.
              dropout_p: Attention dropout rate.
              rows: Query rows sharing K/V, or ``-1`` to use the key count.
              **kwargs: The rest of the owner's bus, unread.

            Returns:
              cost: Whole-invocation FLOPs and logical tensor bytes.

            Raises:
              ValueError: ``num_heads_kv`` exceeds ``num_heads`` or does not
                divide it.

            """
            # A dense mask does not remove rows or columns from either product.
            del kwargs, window
            return attention_kernel_cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                num_heads=num_heads,
                num_heads_kv=num_heads_kv,
                channels_head=channels_head,
                channels_v_head=channels_v_head,
                window=-1,
                dropout_p=dropout_p,
                rows=rows,
            )

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
        window: int = -1,
        scale: float | None = None,
        **kwargs: object,
    ) -> Tensor:
        del kwargs
        attn_mask, is_causal = combined_mask(
            q,
            k,
            is_causal=is_causal,
            attn_mask=attn_mask,
            window=window,
        )
        if is_causal and attn_mask is None and q.shape[-3] != k.shape[-3]:
            # SDPA's ``is_causal=True`` is top-left aligned when query and
            # key lengths differ, which is wrong for cached decode (the
            # query chunk sits at the end of the key cache), so an explicit
            # bottom-right causal mask is built instead.
            attn_mask = causal_chunk_mask(q, k)
        q, k, v = (t.movedim(-3, -2) for t in (q, k, v))
        out = functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            # SDPA refuses an explicit mask together with is_causal.
            is_causal=is_causal and attn_mask is None,
            scale=scale,
        )
        return out.movedim(-3, -2)


class SdpaNaive(nn.Module):
    """Manual matmul+softmax attention (matches HF eager_attention_forward)."""

    class Config(Fig["SdpaNaive"]):
        @classmethod
        def cost(
            cls,
            *,
            seq_len: int,
            batch_size: int = 1,
            dtype: torch.dtype | None,
            num_heads: int,
            channels_head: int,
            channels_v_head: int = -1,
            num_heads_kv: int = -1,
            window: int = -1,
            dropout_p: float = 0.0,
            rows: int = -1,
            **kwargs: object,
        ) -> Cost:
            """Cost the kernel from the shapes its owner hands it.

            See :func:`attention_kernel_cost` for every argument.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              num_heads: Query heads.
              num_heads_kv: Key/value heads; -1 mirrors num_heads.
              channels_head: Width of each query/key head.
              channels_v_head: Value width; -1 uses the query/key width.
              window: Previous keys each query reaches, plus itself; negative is unbounded.
              dropout_p: Attention dropout rate.
              rows: Query rows sharing K/V, or ``-1`` to use the key count.
              **kwargs: The rest of the owner's bus, unread.

            Returns:
              cost: Whole-invocation FLOPs and logical tensor bytes.

            Raises:
              ValueError: ``num_heads_kv`` exceeds ``num_heads`` or does not
                divide it.

            """
            # A dense mask does not remove rows or columns from either product.
            del kwargs, window
            return attention_kernel_cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                num_heads=num_heads,
                num_heads_kv=num_heads_kv,
                channels_head=channels_head,
                channels_v_head=channels_v_head,
                window=-1,
                dropout_p=dropout_p,
                rows=rows,
            )

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
        window: int = -1,
        scale: float | None = None,
        **kwargs: object,
    ) -> Tensor:
        del kwargs
        attn_mask, is_causal = combined_mask(
            q,
            k,
            is_causal=is_causal,
            attn_mask=attn_mask,
            window=window,
        )
        q, k, v = (t.movedim(-3, -2) for t in (q, k, v))
        # A separate name, not a rebind: ``q`` comes back from the generator
        # unpacking above partially unknown, so assigning into the ``float |
        # None`` parameter widens it rather than narrowing it.
        logit_scale = q.shape[-1] ** -0.5 if scale is None else scale
        attn = torch.matmul(q, k.transpose(-2, -1)) * logit_scale
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
            attn = functional.dropout(attn, p=dropout_p)
        return torch.matmul(attn, v).movedim(-3, -2)


class SdpaVarlen(nn.Module):
    """Causal SDPA within packed segments under a dense mask: the portable varlen kernel.

    The layout is ``SdpaFused``'s, ``[..., S, num_heads, channels_head]``, and
    ``cu_seqlens`` bounds the segments of the flattened leading and ``S`` axes,
    none crossing a row (:func:`~priml.model.attention.window.segment_mask`).
    Keys and values may carry fewer heads than the queries, a divisor of
    theirs, which SDPA groups. It runs on any device and is the reference a
    varlen flash kernel must match, at the dense mask's ``S × S`` memory per row.

    Attributes:
      max_logit: The largest scaled ``q·k`` the mask admitted in the last
        forward that asked for it (``record_max_logit``), detached float32;
        None otherwise.

    """

    class Config(Fig["SdpaVarlen"]):
        """No options: the segments come from ``cu_seqlens`` at call time."""

        @classmethod
        def cost(
            cls,
            *,
            seq_len: int,
            batch_size: int = 1,
            dtype: torch.dtype | None,
            num_heads: int,
            channels_head: int,
            channels_v_head: int = -1,
            num_heads_kv: int = -1,
            window: int = -1,
            dropout_p: float = 0.0,
            rows: int = -1,
            **kwargs: object,
        ) -> Cost:
            """Cost the kernel from the shapes its owner hands it.

            See :func:`attention_kernel_cost` for every argument.

            Args:
              seq_len: Tokens per row.
              batch_size: Rows in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              num_heads: Query heads.
              num_heads_kv: Key/value heads; -1 mirrors num_heads.
              channels_head: Width of each query/key head.
              channels_v_head: Value width; -1 uses the query/key width.
              window: Previous keys each query reaches, plus itself; negative is unbounded.
              dropout_p: Attention dropout rate.
              rows: Query rows sharing K/V, or ``-1`` to use the key count.
              **kwargs: The rest of the owner's bus, unread.

            Returns:
              cost: Whole-invocation FLOPs and logical tensor bytes.

            Raises:
              ValueError: ``num_heads_kv`` exceeds ``num_heads`` or does not
                divide it.

            """
            # A dense mask, segments or window, removes no rows or columns.
            del kwargs, window
            return attention_kernel_cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                num_heads=num_heads,
                num_heads_kv=num_heads_kv,
                channels_head=channels_head,
                channels_v_head=channels_v_head,
                window=-1,
                dropout_p=dropout_p,
                rows=rows,
            )

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()
        self.max_logit: Tensor | None = None

    @override
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        cu_seqlens: Tensor,
        window: int = -1,
        is_causal: bool = True,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        scale: float | None = None,
        record_max_logit: bool = False,
        **kwargs: object,
    ) -> Tensor:
        """Attend causally within each segment.

        Args:
          q: Queries ``[..., S, H, D]``.
          k: Keys ``[..., S, H_kv, D]``; ``H_kv`` divides ``H``.
          v: Values, shaped like ``k``.
          cu_seqlens: Int32 segment boundaries of the flattened leading and
            ``S`` axes.
          window: Previous keys each query reaches, plus itself; -1 for all.
          is_causal: Must be True: segments attend causally.
          attn_mask: Must be None; the segments are the mask.
          dropout_p: Must be 0.
          scale: Logit scale; None is ``D**-0.5``.
          record_max_logit: Set ``max_logit``; it forms the logits a second
            time, ``H`` times the mask's memory.
          **kwargs: The rest of the bus, unread.

        Returns:
          out: ``[..., S, H, D]``.

        Raises:
          ValueError: Non-causal attention, a mask, or dropout was asked for.

        """
        del kwargs
        if not is_causal or attn_mask is not None or dropout_p:
            raise ValueError("SdpaVarlen is causal and takes no mask or dropout.")
        lead, length = q.shape[:-3], q.shape[-3]
        mask = segment_mask(
            cu_seqlens,
            rows=math.prod(lead),
            length=length,
            window=window,
        ).view(*lead, 1, length, length)
        q, k, v = (t.movedim(-3, -2) for t in (q, k, v))
        out = functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            scale=scale,
            enable_gqa=True,
        )
        self.max_logit = None
        if record_max_logit:
            # SDPA never exposes its logits, so they are formed again.
            with torch.no_grad():
                keys = k.repeat_interleave(q.shape[-3] // k.shape[-3], dim=-3)
                logit_scale = float(q.shape[-1] ** -0.5) if scale is None else scale
                logits = q @ keys.transpose(-1, -2) * logit_scale
                masked = logits.masked_fill(~mask, float("-inf"))
                self.max_logit = masked.amax().float()
        return out.movedim(-2, -3)
