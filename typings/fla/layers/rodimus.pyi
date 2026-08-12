from typing import Any
from fla.models.utils import Cache
from torch import nn
from transformers.processing_utils import Unpack

import torch

logger = ...

def align_multiple(value, multiple_size=...): ...
def autocast_to_fp16(x): ...

class RodimusAttention(nn.Module):
    def __init__(
        self,
        block_type: str = ...,
        mode: str = ...,
        hidden_size: int = ...,
        input_gate_low_rank: float | str | None = ...,
        expand_ratio: int = ...,
        use_short_conv: bool = ...,
        conv_size: int = ...,
        conv_bias: bool = ...,
        norm_eps: float = ...,
        k_norm_eps: float | None = ...,
        residual_in_fp32: bool = ...,
        layer_idx: int = ...,
    ) -> None: ...
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

class SlidingWindowSharedKeyAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = ...,
        num_heads: int = ...,
        qkv_bias: bool = ...,
        qk_norm: bool = ...,
        window_size: int = ...,
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
