from fla.utils import autotune_cache_kwargs

import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "STORE_QG": lambda args: args["qg"] is not None,
        "STORE_KG": lambda args: args["kg"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "HV", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kda_kernel(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "HV", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def prepare_wy_repr_bwd_kda_kernel(
    k,
    v,
    beta,
    gk,
    A,
    dA,
    dw,
    du,
    dk,
    dk2,
    dv,
    db,
    dg,
    dg2,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
    q: torch.Tensor | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]: ...
def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gk: torch.Tensor,
    A: torch.Tensor,
    dk: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
