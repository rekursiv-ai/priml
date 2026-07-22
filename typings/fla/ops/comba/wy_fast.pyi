from fla.utils import autotune_cache_kwargs

import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_G": lambda args: args["g"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "BT", "IS_VARLEN", "USE_G"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_comba_pkt_fwd_kernel(
    k,
    p,
    beta,
    g0,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    USE_EXP2: tl.constexpr,
): ...
def chunk_scaled_dot_comba_pkt_fwd(
    k: torch.Tensor,
    p: torch.Tensor,
    beta: torch.Tensor,
    g0: torch.Tensor | None = ...,
    g: torch.Tensor | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    output_dtype: torch.dtype = ...,
    chunk_indices: torch.LongTensor | None = ...,
    use_exp2: bool = ...,
) -> torch.Tensor: ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def prepare_wy_repr_bwd_kernel(
    k,
    v,
    p,
    beta,
    g0,
    g,
    A,
    dw,
    du,
    dk,
    dv,
    dp,
    dbeta,
    dg0,
    dg,
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
    USE_EXP2: tl.constexpr,
): ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
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
    USE_EXP2: tl.constexpr,
): ...
def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None = ...,
    use_exp2: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g0: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None = ...,
    use_exp2: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
