from typing import Any

from torch import nn

import torch

"""
https://github.com/corl-team/rebased/blob/main/flash_linear_attention/fla/layers/rebased_fast.py
"""

class ReBasedLinearAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        l_max: int = ...,
        feature_dim: int = ...,
        num_key_value_heads: int = ...,
        num_heads: int = ...,
        use_gamma: bool | None = ...,
        use_beta: bool | None = ...,
        normalize: bool | None = ...,
        causal: bool = ...,
        eps: float = ...,
        mode: str = ...,
        layer_idx: int | None = ...,
        **kwargs,
    ) -> ReBasedLinearAttention: ...
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def forward_reference(
        self, hidden_states: torch.Tensor, filters: torch.Tensor = ..., *args, **kwargs
    ) -> Any: ...
