from fla.models.utils import Cache
from fla.ops.log_linear_attn.chunk import LogLinearAttentionState
from torch import nn

import torch

logger = ...

def ceil_log(x: int, b: int) -> int: ...
def get_num_levels(length: int, base: int) -> int: ...

MAX_SEQUENCE_LENGTH = ...
LAMBDA_LEVEL_BASE = ...
MAX_NUM_LEVELS = ...

def hmamba_chunk_scan_combined(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dl: torch.Tensor,
    L: torch.Tensor,
    chunk_size: int,
    D: torch.Tensor | None = ...,
    z: torch.Tensor | None = ...,
    dt_bias: torch.Tensor | None = ...,
    initial_states: LogLinearAttentionState | None = ...,
    seq_idx: torch.Tensor | None = ...,
    cu_seqlens: torch.Tensor | None = ...,
    dt_softplus: bool = ...,
    dt_limit: tuple[float, float] = ...,
    return_final_states: bool = ...,
) -> tuple[Tensor, Tensor]: ...
def hmamba_split_conv1d_scan_combined(
    zxbcdtdl: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    L: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int,
    initial_states: torch.Tensor | None = ...,
    seq_idx: torch.Tensor | None = ...,
    dt_limit: tuple[float, float] = ...,
    return_final_states: bool = ...,
    activation: str = ...,
    rmsnorm_weight: torch.Tensor | None = ...,
    rmsnorm_eps: float = ...,
    outproj_weight: torch.Tensor | None = ...,
    outproj_bias: torch.Tensor | None = ...,
    headdim: int | None = ...,
    ngroups: int = ...,
    norm_before_gate: bool = ...,
    conv1d_fn=...,
    conv_backend: str = ...,
) -> torch.Tensor: ...

class LogLinearMamba2(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int = ...,
        hidden_size: int = ...,
        state_size: int = ...,
        expand: int = ...,
        n_groups: int = ...,
        conv_kernel: int = ...,
        conv_init: float | None = ...,
        use_conv_bias: bool = ...,
        hidden_act: str = ...,
        rmsnorm: bool = ...,
        D_has_hdim: bool = ...,
        norm_before_gate: bool = ...,
        chunk_size: int = ...,
        dt_limit: tuple[float, float] = ...,
        dt_min: float = ...,
        dt_max: float = ...,
        dt_init_floor: float = ...,
        use_bias: bool = ...,
        norm_eps: float = ...,
        layer_idx: int = ...,
        backend: str = ...,
    ) -> None: ...
    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        last_state: dict | None = ...,
        use_cache: bool = ...,
        attention_mask: torch.Tensor | None = ...,
    ) -> (
        tuple[Any, Any, Any] | tuple[Any, None, None] | tuple[Any, Tensor | None, Any]
    ): ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...
