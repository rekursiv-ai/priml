from typing import Any
from fla.models.utils import Cache
from torch import nn
from transformers.processing_utils import Unpack

import torch

class GatedSlotAttention(nn.Module):
    def __init__(
        self,
        mode: str = ...,
        hidden_size: int = ...,
        expand_k: float = ...,
        expand_v: float = ...,
        num_heads: int = ...,
        num_kv_heads: int | None = ...,
        use_short_conv: bool = ...,
        conv_size: int = ...,
        conv_bias: bool = ...,
        num_slots: int | None = ...,
        elementwise_affine: bool | None = ...,
        norm_eps: float = ...,
        gate_logit_normalizer: int = ...,
        feature_map: str = ...,
        use_output_gate: bool = ...,
        use_norm: bool = ...,
        layer_idx: int | None = ...,
        scale: float | None = ...,
        **kwargs,
    ) -> GatedSlotAttention: ...
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
