from typing import Any

from fla.models.utils import Cache
from torch import nn
from transformers.processing_utils import Unpack

import torch

class GatedDeltaProduct(nn.Module):
    def __init__(
        self,
        hidden_size: int = ...,
        expand_v: float = ...,
        head_dim: int = ...,
        num_heads: int = ...,
        num_v_heads: int = ...,
        mode: str = ...,
        use_output_gate: bool = ...,
        use_short_conv: bool = ...,
        conv_size: int = ...,
        conv_bias: bool = ...,
        layer_idx: int = ...,
        norm_eps: float = ...,
        use_forget_gate: bool = ...,
        allow_neg_eigval: bool = ...,
        num_householder: int = ...,
        **kwargs,
    ) -> GatedDeltaProduct: ...
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
