from typing import Any
from fla.models.utils import Cache
from torch import nn

import torch

""" Implementing the Deepseek Multi Latent Attention (MLA) module. Reference:

https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L328
"""
logger = ...

def yarn_get_mscale(scale=..., mscale=...) -> float: ...

class MultiheadLatentAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = ...,
        num_heads: int = ...,
        q_lora_rank: int | None = ...,
        qk_rope_head_dim: int = ...,
        kv_lora_rank: int = ...,
        v_head_dim: int = ...,
        qk_nope_head_dim: int = ...,
        qk_head_dim: int | None = ...,
        window_size: int | None = ...,
        rope_theta: float = ...,
        max_position_embeddings: int | None = ...,
        rope_scaling: dict | None = ...,
        layer_idx: int = ...,
    ) -> MultiheadLatentAttention: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...
