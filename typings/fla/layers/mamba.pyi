from typing import Any

from fla.models.utils import Cache
from torch import Tensor, nn
from transformers.processing_utils import Unpack

import torch

logger = ...

class Mamba(nn.Module):
    def __init__(
        self,
        hidden_size: int = ...,
        state_size: int = ...,
        conv_kernel: int = ...,
        use_conv_bias: bool = ...,
        intermediate_size: int = ...,
        dt_rank: int | str = ...,
        dt_min: float = ...,
        dt_max: float = ...,
        dt_init: str = ...,
        dt_scale: float = ...,
        dt_init_floor: float = ...,
        use_bias: bool = ...,
        hidden_act: str = ...,
        layer_idx: int = ...,
        backend: str = ...,
    ) -> None: ...
    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        last_state: dict | None = ...,
        use_cache: bool | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        **kwargs: Unpack[dict],
    ) -> tuple[Any, None, None] | tuple[Any, Any | Tensor | None, Any | None]: ...
    def slow_forward(
        self,
        input_states,
        last_state: dict | None = ...,
        use_cache: bool | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        **kwargs: Unpack[dict],
    ) -> tuple[Any, Any | Tensor | None, Any | Tensor]: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...
