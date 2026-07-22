from fla.utils import USE_CUDA_GRAPH, autotune_cache_kwargs

import torch
import triton
import triton.language as tl

NUM_WARPS_AUTOTUNE = ...
BK_LIST = ...

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BK_LIST
        for BV in BK_LIST
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
    ],
    key=["BT"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_dplr_fwd_kernel_o(
    qg,
    v,
    v_new,
    A_qk,
    A_qb,
    h,
    o,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def chunk_dplr_fwd_o(
    qg: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    A_qk: torch.Tensor,
    A_qb: torch.Tensor,
    h: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> torch.Tensor: ...
