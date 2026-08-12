from typing import Any
from fla.models.utils import Cache
from torch import nn

import torch

logger = ...

class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int = ...,
        num_heads: int = ...,
        num_kv_heads: int | None = ...,
        qkv_bias: bool = ...,
        qk_norm: bool = ...,
        window_size: int | None = ...,
        rope_theta: float | None = ...,
        max_position_embeddings: int | None = ...,
        layer_idx: int = ...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...
