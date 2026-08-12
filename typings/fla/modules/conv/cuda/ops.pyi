from torch import Tensor
from typing import Any
from fla.utils import input_guard

import torch

"""CUDA-based mixed-mode implementation for causal convolution."""

class FastCausalConv1dFn(torch.autograd.Function):
    @staticmethod
    @input_guard(no_guard_contiguous=["x"])
    def forward(
        ctx,
        x,
        weight,
        bias=...,
        residual: torch.Tensor | None = ...,
        initial_states=...,
        output_final_state=...,
        activation=...,
        cu_seqlens: torch.LongTensor | None = ...,
        cu_seqlens_cpu: torch.LongTensor | None = ...,
        chunk_indices: torch.LongTensor | None = ...,
        seq_idx: torch.LongTensor | None = ...,
    ) -> tuple[Any, None]: ...
    @staticmethod
    @input_guard
    def backward(
        ctx, dout, *args
    ) -> tuple[
        Any, Any, Any | None, None, None, None, None, None, None, None, None
    ]: ...

def fast_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor | None = ...,
    bias: torch.Tensor | None = ...,
    residual: torch.Tensor | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool | None = ...,
    activation: str | None = ...,
    cu_seqlens: torch.Tensor | None = ...,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
    seq_idx: torch.LongTensor | None = ...,
) -> None: ...
def causal_conv1d_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = ...,
    residual: torch.Tensor | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool | None = ...,
    activation: str | None = ...,
    cu_seqlens: torch.Tensor | None = ...,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
    **kwargs,
) -> tuple[Any, Any | Tensor | None]: ...
