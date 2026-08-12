from typing import Any
from fla.models.utils import Cache
from torch import nn

import torch

class RWKV7Attention(nn.Module):
    def __init__(
        self,
        mode: str = ...,
        hidden_size: int = ...,
        head_dim: int | None = ...,
        num_heads: int | None = ...,
        decay_low_rank_dim: int | None = ...,
        gate_low_rank_dim: int | None = ...,
        a_low_rank_dim: int | None = ...,
        v_low_rank_dim: int | None = ...,
        elementwise_affine: bool | None = ...,
        norm_eps: float = ...,
        layer_idx: int = ...,
        fuse_norm: bool = ...,
        value_dim: int = ...,
        num_hidden_layers: int = ...,
        **kwargs,
    ) -> RWKV7Attention: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        v_first: torch.Tensor = ...,
        cu_seqlens: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...
