from typing import Any

from fla.models.utils import Cache
from torch import Tensor, nn
from transformers.processing_utils import Unpack

import torch

def transform(
    x: torch.Tensor,
    routing_mask: torch.Tensor,
    num_memories: int,
    selected_memories: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[Any, Tensor, Tensor, Tensor, Any, Any]: ...
def reconstruct(
    transformed_x,
    indices: torch.Tensor,
    sorted_indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
    topk: int,
    routing_weights: torch.Tensor,
    mask: torch.Tensor,
) -> Tensor: ...

class MomAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = ...,
        head_dim: int = ...,
        num_heads: int = ...,
        expand_v: float = ...,
        mode: str = ...,
        use_output_gate: bool = ...,
        use_short_conv: bool = ...,
        conv_size: int = ...,
        conv_bias: bool = ...,
        layer_idx: int = ...,
        norm_eps: float = ...,
        num_memories: int = ...,
        topk: int = ...,
        capacity: float = ...,
        shared_mem: bool = ...,
        single_kv_proj: bool = ...,
        **kwargs,
    ) -> MomAttention: ...
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
    def shared_o(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        recurrent_state=...,
        use_cache: bool | None = ...,
        conv_state_q=...,
        conv_state_k=...,
        conv_state_v=...,
        **kwargs,
    ) -> torch.Tensor: ...
    def cu2pad(self, x, cu_seqlens) -> tuple[Tensor, Tensor]: ...
    def pad_for_conv(
        self, cu_seqlens, cu_q, cu_k, cu_v
    ) -> tuple[Tensor, Any, Any, Any, Tensor]: ...
    def unpad_after_conv(
        self, conv_cu_seqlens, cu_seqlens, cu_q, cu_k, cu_v, pad_lengths
    ) -> tuple[Tensor, Tensor, Tensor]: ...
    def prepare_recurrent_state(
        self, recurrent_state, cu_seqlens, cu_seqlen_all, reverse_indices, batch_size
    ) -> Tensor | None: ...
    def handle_recurrent_state(
        self,
        recurrent_state,
        recurrent_state_new,
        cu_seqlens,
        cu_seqlen_all,
        reverse_indices,
    ) -> Tensor | None: ...
