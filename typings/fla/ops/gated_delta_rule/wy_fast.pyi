from fla.utils import IS_NVIDIA_BLACKWELL, autotune_cache_kwargs

import torch
import triton
import triton.language as tl

if IS_NVIDIA_BLACKWELL:
    @triton.jit
    def safe_dot(a, b) -> tensor | tuple: ...

else:
    @triton.jit
    def safe_dot(a, b): ...

@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
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
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
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
def prepare_wy_repr_bwd_kernel(
    k,
    v,
    beta,
    g,
    A,
    dw,
    du,
    dk,
    dv,
    db,
    dg,
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
    USE_G: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
    use_exp2: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    g: torch.Tensor = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
    use_exp2: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

fwd_recompute_w_u = ...
bwd_prepare_wy_repr = ...
