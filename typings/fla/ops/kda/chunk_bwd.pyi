from typing import Any

from fla.ops.backends import dispatch
from fla.ops.cp import FLACPContext
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs
from torch import Tensor

import torch
import triton
import triton.language as tl

BK_LIST = ...
BV_LIST = ...
NUM_WARPS = ...

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=["H", "HV", "K", "V", "BT", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_bwd_kernel_dAv(
    q,
    k,
    v,
    A,
    do,
    dv,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
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
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BK_LIST
        for BV in BV_LIST
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
        if not (IS_NVIDIA_HOPPER and BK == 32 and num_warps == 4)
    ],
    key=["BT", "HV", "TRANSPOSE_STATE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_bwd_kernel_wy_dqkg_fused(
    q,
    k,
    v,
    v_new,
    g,
    beta,
    A,
    h,
    do,
    dh,
    dq,
    dk,
    dv,
    dv2,
    dg,
    db,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def chunk_kda_bwd_dAv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    A: torch.Tensor | None = ...,
    scale: float = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
@dispatch("kda")
def chunk_kda_bwd_wy_dqkg_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    chunk_indices: torch.LongTensor | None = ...,
    transpose_state_layout: bool = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: ...
def chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    g: torch.Tensor | None = ...,
    g_org: torch.Tensor | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    safe_gate: bool = ...,
    lower_bound: float | None = ...,
    use_gate_in_kernel: bool = ...,
    A_log: torch.Tensor | None = ...,
    dt_bias: torch.Tensor | None = ...,
    disable_recompute: bool = ...,
    cp_context: FLACPContext | None = ...,
    transpose_state_layout: bool = ...,
    **kwargs,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Any | Tensor,
    Tensor | Any,
    Tensor,
    Tensor | None,
    Tensor | None,
]: ...
