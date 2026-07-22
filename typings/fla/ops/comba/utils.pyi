from fla.utils import autotune_cache_kwargs

import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["B", "H", "BT", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_comba_cumsum_scalar_fwd_kernel(
    g,
    g0,
    g1,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def chunk_comba_cumsum_scalar_fwd(
    g: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = ...,
    output_dtype: torch.dtype | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
    scale: float | None = ...,
) -> torch.Tensor: ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["B", "H", "BT", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_comba_cumsum_scalar_bwd_kernel(
    dg0,
    dgr,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def chunk_comba_cumsum_scalar_bwd(
    dg0: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = ...,
    output_dtype: torch.dtype | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> torch.Tensor: ...
