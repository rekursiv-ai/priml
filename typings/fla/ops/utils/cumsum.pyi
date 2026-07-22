from fla.utils import autotune_cache_kwargs, input_guard

import torch
import triton
import triton.language as tl

BS_LIST = ...

@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["B", "H", "BT", "IS_VARLEN", "REVERSE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
): ...
@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=["B", "H", "S", "BT", "IS_VARLEN", "REVERSE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
): ...
@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in [32, 64, 128, 256]
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["B", "H", "IS_VARLEN", "REVERSE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_global_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
): ...
@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in [16, 32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["B", "H", "S", "IS_VARLEN", "REVERSE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_global_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
): ...
def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = ...,
    scale: float = ...,
    cu_seqlens: torch.Tensor | None = ...,
    head_first: bool = ...,
    output_dtype: torch.dtype | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> torch.Tensor: ...
def chunk_local_cumsum_vector(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = ...,
    scale: float = ...,
    cu_seqlens: torch.Tensor | None = ...,
    head_first: bool = ...,
    output_dtype: torch.dtype | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> torch.Tensor: ...
@input_guard
def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    reverse: bool = ...,
    cu_seqlens: torch.Tensor | None = ...,
    scale: float = ...,
    head_first: bool = ...,
    output_dtype: torch.dtype | None = ...,
) -> torch.Tensor: ...
@input_guard
def chunk_global_cumsum_vector(
    s: torch.Tensor,
    reverse: bool = ...,
    cu_seqlens: torch.Tensor | None = ...,
    scale: float = ...,
    head_first: bool = ...,
    output_dtype: torch.dtype | None = ...,
) -> torch.Tensor: ...
@input_guard
def chunk_global_cumsum(
    s: torch.Tensor,
    reverse: bool = ...,
    cu_seqlens: torch.Tensor | None = ...,
    scale: float = ...,
    head_first: bool = ...,
    output_dtype: torch.dtype | None = ...,
) -> torch.Tensor: ...
@input_guard
def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = ...,
    scale: float = ...,
    cu_seqlens: torch.Tensor | None = ...,
    head_first: bool = ...,
    output_dtype: torch.dtype | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
    **kwargs,
) -> torch.Tensor: ...
