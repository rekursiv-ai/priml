from typing import Any

from fla.models.utils import Cache
from torch import nn

import torch

class RWKV6Attention(nn.Module):
    def __init__(
        self,
        mode: str = ...,
        hidden_size: int = ...,
        expand_k: float = ...,
        expand_v: float = ...,
        num_heads: int = ...,
        gate_fn: str = ...,
        proj_low_rank_dim: int = ...,
        gate_low_rank_dim: int = ...,
        fuse_norm: bool = ...,
        elementwise_affine: bool | None = ...,
        norm_eps: float = ...,
        layer_idx: int = ...,
        **kwargs,
    ) -> RWKV6Attention: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        cu_seqlens: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...

class LoRA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: bool | None = ...,
        activation: str | None = ...,
    ) -> None: ...
    def set_bias_value(self, value) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class LerpLinear(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, low_rank_dim: int | None = ...
    ) -> None: ...
    def forward(
        self,
        x: torch.Tensor,
        delta: torch.Tensor | None = ...,
        cu_seqlens: torch.LongTensor | None = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class DDLerpLinear(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, low_rank_dim: int | None = ...
    ) -> None: ...
    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        delta: torch.Tensor | None = ...,
        cu_seqlens: torch.LongTensor | None = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
