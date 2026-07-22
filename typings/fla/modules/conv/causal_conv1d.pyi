from fla.ops.cp import FLACPContext
from fla.utils import input_guard

import torch

"""Main interface for causal 1D convolution operations."""

@input_guard(no_guard_contiguous=["x"])
def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor | None = ...,
    bias: torch.Tensor | None = ...,
    residual: torch.Tensor | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool | None = ...,
    activation: str | None = ...,
    backend: str | None = ...,
    cu_seqlens: torch.Tensor | None = ...,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
    cp_context: FLACPContext | None = ...,
    **kwargs,
) -> (
    tuple[Any | None, None]
    | tuple[Any, Any]
    | Any
    | tuple[Any, Any | Tensor | None]
    | None
): ...
