"""Attention kernel implementations."""

from __future__ import annotations

from typing import override

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
from priml.model.attention.window import causal_chunk_mask, combined_mask


def attention_kernel_cost(
    *,
    seq_len: int,
    batch_size: int = 1,
    dtype: torch.dtype | None,
    num_heads: int,
    channels_head: int,
    channels_v_head: int = -1,
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
      channels_head: Width of each query/key head.
      channels_v_head: Value width; -1 uses the query/key width.
      window: Previous keys each query reaches, plus itself; negative is unbounded.
      dropout_p: Attention dropout rate; nonzero adds a mask and a rescale.
      rows: Query rows sharing K/V, or ``-1`` to use the key count.
      **kwargs: The rest of the owner's bus, unread.

    Returns:
      cost: Unfused logical tensor I/O and FLOPs, not measured HBM traffic.
        Fused and naive kernels share this algorithmic accounting convention.

    """
    del kwargs
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
    ).tile(batch_size)
    values = matmul_cost(
        channels_in=keys,
        channels_out=value_width,
        weight=False,
        rows=sequence_rows,
        dtype=dt,
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
    return (scores + values + softmax + dropout).tile(num_heads, copies=num_heads)


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
              channels_head: Width of each query/key head.
              channels_v_head: Value width; -1 uses the query/key width.
              window: Previous keys each query reaches, plus itself; negative is unbounded.
              dropout_p: Attention dropout rate.
              rows: Query rows sharing K/V, or ``-1`` to use the key count.
              **kwargs: The rest of the owner's bus, unread.

            Returns:
              cost: Whole-invocation FLOPs and logical tensor bytes.

            """
            # A dense mask does not remove rows or columns from either product.
            del kwargs, window
            return attention_kernel_cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                num_heads=num_heads,
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
              channels_head: Width of each query/key head.
              channels_v_head: Value width; -1 uses the query/key width.
              window: Previous keys each query reaches, plus itself; negative is unbounded.
              dropout_p: Attention dropout rate.
              rows: Query rows sharing K/V, or ``-1`` to use the key count.
              **kwargs: The rest of the owner's bus, unread.

            Returns:
              cost: Whole-invocation FLOPs and logical tensor bytes.

            """
            # A dense mask does not remove rows or columns from either product.
            del kwargs, window
            return attention_kernel_cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                num_heads=num_heads,
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
