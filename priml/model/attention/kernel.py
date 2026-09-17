"""Attention kernel implementations."""

from __future__ import annotations

from typing import override

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional as f

import torch

from priml.model.attention.window import window_mask
from priml.model.cost import (
    Bytes,
    Compute,
    Cost,
    Flops,
    elementwise_cost,
    matmul_cost,
    reduction_cost,
)


def attention_kernel_cost(
    config: object,
    *,
    seq_len: int,
    num_heads: int,
    channels_head: int,
    channels_v_head: int = -1,
    window: int = -1,
    dropout_p: float = 0.0,
    rows: int = 1,
    itemsize: int = 4,
    **kwargs: object,
) -> Cost:
    """Price ``softmax(QK^T)V`` for one query row across every head.

    Bound as ``Config.cost`` on every kernel whose products are the standard
    two, so ``config`` is the kernel config and unread: a kernel config holds
    no fields. The shapes arrive on the message bus the way ``q``/``k``/``v``
    arrive in ``forward``: the OWNER of the projections says how many heads
    and how wide. Counted over the full window with no causal discount, per
    the module policy.

    Args:
      config: The kernel config; carries nothing this needs.
      seq_len: Keys before any window.
      num_heads: Query heads.
      channels_head: Width of each query/key head.
      channels_v_head: Value width; -1 uses the query/key width.
      rows: Batch rows; activation reuse is limited to one full window.
      itemsize: Uniform bytes per tensor element.
      window: Keys each query reaches, or ``-1`` for the whole sequence.
      dropout_p: Attention dropout rate; nonzero adds a mask and a rescale.
      **kwargs: The rest of the bus, ignored here.

    Returns:
      cost: Unfused logical tensor I/O and FLOPs, not measured HBM traffic.
        Fused and naive kernels share this algorithmic accounting convention.

    """
    del config, rows, kwargs
    keys = seq_len if window < 0 else min(window, seq_len)
    value_width = channels_head if channels_v_head < 0 else channels_v_head
    # Full-window blocks share K/V across their query rows, never across batches.
    scores = matmul_cost(
        channels_in=channels_head,
        channels_out=keys,
        weight=False,
        rows=keys,
        itemsize=itemsize,
    )
    values = matmul_cost(
        channels_in=keys,
        channels_out=value_width,
        weight=False,
        rows=keys,
        itemsize=itemsize,
    )
    # Logical unfused I/O, including scores, even when execution uses fused SDPA.
    # Scale/subtract/exp/divide read two row scalars; VJP reads one row sum.
    softmax = Cost(
        primal=Compute(
            flops=Flops(elementwise=4 * keys),
            bytes=Bytes(elementwise=itemsize * (8 * keys + 2)),
        )
        + 2 * reduction_cost(input_elements=keys, itemsize=itemsize),
        adjoint=Compute(
            flops=Flops(elementwise=4 * keys),
            bytes=Bytes(elementwise=itemsize * (10 * keys + 1)),
        )
        + reduction_cost(input_elements=keys, itemsize=itemsize),
    )
    dropout = elementwise_cost(
        primal=2 * keys if dropout_p > 0 else 0,
        adjoint=2 * keys if dropout_p > 0 else 0,
        channels=keys if dropout_p > 0 else 0,
        inputs=3,
        outputs=2,
        adjoint_inputs=2,
        rows=keys,
        itemsize=itemsize,
    )
    return num_heads * (scores + values + softmax + dropout)


class SdpaFused(nn.Module):
    """Wraps F.scaled_dot_product_attention.

    Takes ``[..., S, num_heads, channels_head]`` -- the layout every priml
    projection emits and the one a fused kernel wants -- and transposes to
    SDPA's ``[..., num_heads, S, channels_head]`` internally. The transpose is a
    stride view rather than a copy, so a kernel that needs the other layout
    (FlashAttention-3) is a drop-in value in the same slot and pays nothing.
    """

    class Config(Fig["SdpaFused"]):
        cost = attention_kernel_cost

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
        if attn_mask is None:
            attn_mask = window_mask(q, k, window=window)
        q, k, v = (t.movedim(-3, -2) for t in (q, k, v))
        out = f.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            # The window mask is already causal, and SDPA REFUSES both at once.
            is_causal=is_causal and attn_mask is None,
            scale=scale,
        )
        return out.movedim(-3, -2)


class SdpaNaive(nn.Module):
    """Manual matmul+softmax attention (matches HF eager_attention_forward)."""

    class Config(Fig["SdpaNaive"]):
        cost = attention_kernel_cost

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
        if attn_mask is None:
            attn_mask = window_mask(q, k, window=window)
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
            attn = f.dropout(attn, p=dropout_p)
        return torch.matmul(attn, v).movedim(-3, -2)
