from fla.models.utils import Cache
from torch import nn
from transformers.processing_utils import Unpack

import torch

class GatedLinearAttention(nn.Module):
    def __init__(
        self,
        mode: str = ...,
        hidden_size: int = ...,
        expand_k: float = ...,
        expand_v: float = ...,
        num_heads: int = ...,
        num_kv_heads: int | None = ...,
        feature_map: str | None = ...,
        use_short_conv: bool = ...,
        conv_size: int = ...,
        conv_bias: bool = ...,
        use_output_gate: bool = ...,
        gate_fn: str = ...,
        elementwise_affine: bool | None = ...,
        norm_eps: float = ...,
        gate_logit_normalizer: int = ...,
        gate_low_rank_dim: int = ...,
        clamp_min: float | None = ...,
        fuse_norm: bool = ...,
        layer_idx: int = ...,
    ) -> GatedLinearAttention: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...
