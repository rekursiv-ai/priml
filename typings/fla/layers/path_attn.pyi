from fla.models.utils import Cache
from torch import nn

import torch

logger = ...

class PaTHAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = ...,
        num_heads: int = ...,
        num_kv_heads: int | None = ...,
        use_forget_gate: bool = ...,
        use_qk_norm: bool = ...,
        layer_idx: int = ...,
        use_low_rank_w: bool = ...,
        use_w_shortconv: bool = ...,
        conv_size: int = ...,
        conv_bias: bool = ...,
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
