from typing import Any
from torch import Tensor
from fla.models.utils import Cache
from torch import nn
from transformers.processing_utils import Unpack

import torch

def elu_p1(x) -> Tensor: ...
def sum_norm(x): ...

class DeltaNet(nn.Module):
    def __init__(
        self,
        mode: str = ...,
        d_model: int = ...,
        hidden_size: int = ...,
        expand_k: float = ...,
        expand_v: float = ...,
        num_heads: int = ...,
        use_beta: bool = ...,
        use_gate: bool = ...,
        use_short_conv: bool = ...,
        conv_size: int = ...,
        conv_bias: bool = ...,
        allow_neg_eigval: bool = ...,
        layer_idx: int = ...,
        qk_activation: str = ...,
        qk_norm: str = ...,
        norm_eps: float = ...,
        **kwargs,
    ) -> DeltaNet: ...
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
