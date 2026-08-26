"""Attention kernel implementations."""

from __future__ import annotations

from typing import override

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional as f

import torch

from priml.model.attention.window import window_mask


class SdpaFused(nn.Module):
    """Wraps F.scaled_dot_product_attention.

    Takes ``[..., S, num_heads, channels_head]`` -- the layout every priml
    projection emits and the one a fused kernel wants -- and transposes to
    SDPA's ``[..., num_heads, S, channels_head]`` internally. The transpose is a
    stride view rather than a copy, so a kernel that needs the other layout
    (FlashAttention-3) is a drop-in value in the same slot and pays nothing.
    """

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
