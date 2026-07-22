from fla.models.utils import Cache
from torch import nn

import torch

class LinearAttention(nn.Module):
    def __init__(
        self,
        mode: str = ...,
        hidden_size: str = ...,
        expand_k: float = ...,
        expand_v: float = ...,
        num_heads: int = ...,
        num_kv_heads: int | None = ...,
        feature_map: str = ...,
        tie_feature_map_qk: bool = ...,
        output_norm: str = ...,
        norm_q: bool = ...,
        norm_k: bool = ...,
        do_feature_map_norm: bool = ...,
        elementwise_affine: bool = ...,
        norm_eps: float = ...,
        layer_idx: int | None = ...,
        **kwargs,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...
