from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    input_guard,
)

import torch
import triton
import triton.language as tl

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BD": BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def mean_pooling_fwd_kernel(
    x,
    o,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BD": BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def mean_pooling_bwd_kernel(
    do,
    dx,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def mean_pooling_fwd(
    x: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> torch.Tensor: ...
def mean_pooling_bwd(
    do: torch.Tensor,
    batch_size: int,
    seq_len: int,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> torch.Tensor: ...

class MeanPoolingFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx, x: torch.Tensor, chunk_size: int, cu_seqlens: torch.LongTensor | None = ...
    ) -> torch.Tensor: ...
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do) -> tuple[torch.Tensor, None, None]: ...

def mean_pooling(
    x: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None = ...,
    head_first: bool = ...,
) -> torch.Tensor: ...
