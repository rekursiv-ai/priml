from typing import Any

from fla.models.utils import Cache
from torch import nn

import torch

class ABCAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = ...,
        expand_k: float = ...,
        expand_v: float = ...,
        num_heads: int = ...,
        use_short_conv: bool = ...,
        conv_size: int = ...,
        conv_bias: bool = ...,
        num_slots: int | None = ...,
        elementwise_affine: bool | None = ...,
        norm_eps: float = ...,
        gate_low_rank_dim: int = ...,
        gate_logit_normalizer: int = ...,
        use_rope: bool = ...,
        use_input_gate: bool = ...,
        use_output_gate: bool = ...,
        use_norm: bool = ...,
        clamp_min: float | None = ...,
        clamp_max: float | None = ...,
        layer_idx: int | None = ...,
        **kwargs,
    ) -> ABCAttention: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...
    def state_size(self, seq_len: int = ...) -> int: ...
