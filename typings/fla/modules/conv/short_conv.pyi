from typing import Any

from torch import Tensor, nn

import torch

"""Short convolution implementation for efficient causal convolutions."""

class ShortConvolution(nn.Conv1d):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = ...,
        activation: str | None = ...,
        backend: str | None = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
        **kwargs,
    ) -> None: ...
    def extra_repr(self) -> str: ...
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = ...,
        mask: torch.Tensor | None = ...,
        cache: torch.Tensor | None = ...,
        output_final_state: bool = ...,
        cu_seqlens: torch.LongTensor | None = ...,
        chunk_indices: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def step(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        cache: torch.Tensor,
        output_final_state: bool = ...,
        cu_seqlens: torch.LongTensor | None = ...,
    ) -> Tensor | tuple[Any, Tensor]: ...
    @property
    def state_size(self) -> int: ...
