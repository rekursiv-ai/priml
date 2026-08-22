from typing import Any

from torch import nn

import torch

"""
Linear attention in Based.
https://github.com/HazyResearch/zoology/blob/main/zoology/mixers/based.py
"""

class BasedLinearAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        feature_dim: int = ...,
        num_key_value_heads: int = ...,
        num_heads: int = ...,
        feature_name: str = ...,
        eps: float = ...,
        causal: bool = ...,
        mode: str = ...,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def forward_reference(self, hidden_states: torch.Tensor, **kwargs) -> Any: ...
