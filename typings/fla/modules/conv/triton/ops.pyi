from typing import Any

from fla.utils import input_guard
from torch import Tensor

import torch

@input_guard(no_guard_contiguous=["x"])
def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    activation: str | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
    BT: int = ...,
) -> torch.Tensor: ...
def compute_dh0_triton(
    dy: torch.Tensor,
    y: torch.Tensor | None,
    weight: torch.Tensor,
    initial_state: torch.Tensor,
    activation: str | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor: ...
def causal_conv1d_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    dht: torch.Tensor,
    weight: torch.Tensor | None = ...,
    bias: torch.Tensor | None = ...,
    residual: torch.Tensor | None = ...,
    initial_state: torch.Tensor | None = ...,
    activation: str | None = ...,
    cu_seqlens: torch.Tensor | None = ...,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
    BT: int = ...,
) -> tuple[Tensor, Any | None, Any | None, Tensor | None, Tensor | None]: ...
@input_guard(no_guard_contiguous=["x"])
def causal_conv1d_update_states(
    x: torch.Tensor,
    state_len: int,
    initial_state: torch.Tensor | None = ...,
    cu_seqlens: torch.Tensor | None = ...,
) -> torch.Tensor: ...
@input_guard(no_guard_contiguous=["x"])
def causal_conv1d_update(
    x: torch.Tensor,
    cache: torch.Tensor,
    residual: torch.Tensor | None = ...,
    weight: torch.Tensor | None = ...,
    bias: torch.Tensor | None = ...,
    activation: str | None = ...,
) -> torch.Tensor: ...

class CausalConv1dFunction(torch.autograd.Function):
    @staticmethod
    @input_guard(no_guard_contiguous=["x"])
    def forward(
        ctx,
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
        chunk_size: int = ...,
    ) -> tuple[Any, Any]: ...
    @staticmethod
    @input_guard(no_guard_contiguous=["dy"])
    def backward(
        ctx, dy: torch.Tensor, dht: torch.Tensor | None = ...
    ) -> tuple[
        Tensor,
        Any | None,
        Any | None,
        Tensor | None,
        Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]: ...
