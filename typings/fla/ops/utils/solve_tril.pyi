from fla.utils import autotune_cache_kwargs, input_guard

import torch
import triton
import triton.language as tl

FLA_TRIL_PRECISION = ...
DOT_PRECISION_AUTOTUNE_LIST = ...

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config(
            {"DOT_PRECISION": "ieee"}, num_warps=num_warps, num_stages=num_stages
        )
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=["BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def solve_tril_16x16_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
) -> None: ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config(
            {"DOT_PRECISION": DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages
        )
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
        for DOT_PRECISION in DOT_PRECISION_AUTOTUNE_LIST
    ],
    key=["H", "BT", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
) -> None: ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config(
            {"DOT_PRECISION": DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages
        )
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4, 5]
        for DOT_PRECISION in DOT_PRECISION_AUTOTUNE_LIST
    ],
    key=["H", "BT", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
) -> None: ...
@input_guard
def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
    output_dtype: torch.dtype = ...,
) -> torch.Tensor: ...
