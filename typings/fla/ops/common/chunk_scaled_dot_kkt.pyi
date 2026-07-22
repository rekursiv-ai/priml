from fla.utils import autotune_cache_kwargs

import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "HV", "K", "BT", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
): ...
def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = ...,
    beta: torch.Tensor | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    output_dtype: torch.dtype = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> torch.Tensor: ...
