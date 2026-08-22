from typing import Any

from fla.models.utils import Cache
from torch import nn
from transformers.processing_utils import Unpack

import torch

class LightNetAttention(nn.Module):
    def __init__(
        self,
        mode: str = ...,
        hidden_size: int = ...,
        num_heads: int | None = ...,
        expand_ratio: int | None = ...,
        use_short_conv: bool = ...,
        conv_size: int = ...,
        conv_bias: bool = ...,
        gate_low_rank_dim: int = ...,
        elementwise_affine: bool | None = ...,
        norm_eps: float = ...,
        layer_idx: int = ...,
    ) -> LightNetAttention: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...
    def state_size(self, **kwargs) -> int: ...
