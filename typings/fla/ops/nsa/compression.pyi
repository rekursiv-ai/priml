from typing import Any

from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    contiguous,
)
from torch import Tensor

import torch
import triton
import triton.language as tl

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit
def parallel_nsa_compression_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    scale,
    cu_seqlens,
    token_indices,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def parallel_nsa_compression_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    scale,
    cu_seqlens,
    token_indices,
    chunk_offsets,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "TC"])
def parallel_nsa_compression_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    scale,
    T,
    TC,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def parallel_nsa_compression_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    scale: float,
    cu_seqlens: torch.LongTensor | None = ...,
    token_indices: torch.LongTensor | None = ...,
) -> tuple[Tensor, Tensor]: ...
def parallel_nsa_compression_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    block_size: int = ...,
    scale: float = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    token_indices: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Any, Any, Tensor]: ...

class ParallelNSACompressionFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(
        ctx, q, k, v, block_size, scale, cu_seqlens
    ) -> tuple[Tensor, Tensor]: ...
    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, *args) -> tuple[Any, Any, Tensor, None, None, None]: ...

def parallel_nsa_compression(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = ...,
    scale: float = ...,
    cu_seqlens: torch.LongTensor | None = ...,
) -> None: ...
