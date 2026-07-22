from fla.utils import autotune_cache_kwargs

import torch
import triton
import triton.language as tl

NUM_WARPS = ...

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_mesa_net_h_kv_bwd_intra_kernel(
    q_star,
    k,
    v,
    beta,
    h_kv,
    g,
    do,
    dh_kv,
    dq,
    dk_beta,
    dg,
    dv,
    cu_seqlens,
    chunk_indices,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def chunk_mesa_net_h_kv_bwd_intra_fn(
    q_star,
    k,
    v,
    beta,
    h_kv,
    dh_kv,
    g,
    do,
    cu_seqlens,
    chunk_size=...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...
