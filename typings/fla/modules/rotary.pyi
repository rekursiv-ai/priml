from fla.utils import autotune_cache_kwargs, input_guard
from torch import nn

import torch
import triton
import triton.language as tl

NUM_WARPS_AUTOTUNE = ...

def rotate_half(x, interleaved=...) -> Tensor: ...
def rotary_embedding_ref(x, cos, sin, interleaved=...) -> Tensor: ...
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
    ],
    key=["B", "H", "D", "INTERLEAVED"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def rotary_embedding_kernel(
    x,
    cos,
    sin,
    y,
    cu_seqlens,
    chunk_indices,
    seq_offsets,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    TR: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
) -> None: ...
def rotary_embedding_fwdbwd(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: int | torch.Tensor = ...,
    cu_seqlens: torch.Tensor | None = ...,
    interleaved: bool = ...,
    inplace: bool = ...,
    conjugate: bool = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> torch.Tensor: ...

class RotaryEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=...,
        inplace=...,
        seqlen_offsets: int | torch.Tensor = ...,
        cu_seqlens: torch.Tensor | None = ...,
        chunk_indices: torch.LongTensor | None = ...,
    ) -> Tensor: ...
    @staticmethod
    @input_guard
    def backward(
        ctx, do
    ) -> tuple[Tensor, None, None, None, None, None, None, None]: ...

def rotary_embedding(
    x,
    cos,
    sin,
    interleaved=...,
    inplace=...,
    seqlen_offsets: int | torch.Tensor = ...,
    cu_seqlens: torch.Tensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> Any | None: ...

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        base: float = ...,
        scale_base: float | None = ...,
        interleaved: bool = ...,
        pos_idx_in_fp32: bool = ...,
        device: torch.device | None = ...,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset: int | torch.Tensor = ...,
        cu_seqlens: torch.Tensor | None = ...,
        max_seqlen: int | None = ...,
        chunk_indices: torch.LongTensor | None = ...,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
