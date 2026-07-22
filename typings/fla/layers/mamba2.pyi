from fla.models.utils import Cache
from torch import nn

import torch

logger = ...

def apply_mask_to_padding_states(hidden_states, attention_mask): ...
def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int) -> Tensor: ...
def reshape_into_chunks(input_tensor, pad_size, chunk_size) -> Tensor: ...
def segment_sum(input_tensor) -> Tensor: ...

class Mamba2(nn.Module):
    def __init__(
        self,
        num_heads: int | None = ...,
        head_dim: int = ...,
        hidden_size: int = ...,
        state_size: int = ...,
        expand: int = ...,
        n_groups: int = ...,
        conv_kernel: int = ...,
        conv_init: float | None = ...,
        use_conv_bias: bool = ...,
        hidden_act: str = ...,
        A_init_range: tuple[float, float] = ...,
        D_has_hdim: bool = ...,
        rmsnorm: bool = ...,
        norm_before_gate: bool = ...,
        dt_min: float = ...,
        dt_max: float = ...,
        dt_init_floor: float = ...,
        dt_limit: tuple[float, float] = ...,
        use_bias: bool = ...,
        norm_eps: float = ...,
        chunk_size: int = ...,
        layer_idx: int | None = ...,
        backend: str = ...,
    ) -> Mamba2: ...
    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        last_state: dict | None = ...,
        use_cache: bool = ...,
        attention_mask: torch.Tensor | None = ...,
    ) -> (
        tuple[Any, Any, Any] | tuple[Any, None, None] | tuple[Any, Tensor | None, Any]
    ): ...
    def torch_forward(
        self,
        input_states,
        last_state: dict | None = ...,
        use_cache: bool = ...,
        attention_mask: torch.Tensor | None = ...,
    ) -> tuple[Any, Any, Any | Tensor] | tuple[Any, Any | Tensor | None, Tensor]: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...
