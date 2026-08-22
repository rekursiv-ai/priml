from typing import Any

from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    input_guard,
)
from torch import Tensor

import torch
import triton
import triton.language as tl

triton_config = ...
NUM_WARPS = ...

@triton.heuristics(
    {
        "NV": lambda args: triton.cdiv(args["V"], args["BV"]),
        "OUTPUT_ATTENTIONS": lambda args: args["attn"] is not None,
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "BS", "BK", "BV", "USE_G"],
    **autotune_cache_kwargs,
)
@triton.jit
def parallel_simple_gla_fwd_kernel(
    q,
    k,
    v,
    g,
    o,
    attn,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    OUTPUT_ATTENTIONS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
): ...
@triton.jit(do_not_specialize=["T"])
def parallel_simple_gla_bwd_kernel_dq(
    i_t,
    i_k,
    i_v,
    q,
    k,
    v,
    g,
    do,
    dq,
    dg,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
): ...
@triton.jit(do_not_specialize=["T"])
def parallel_simple_gla_bwd_kernel_dkv(
    i_t,
    i_k,
    i_v,
    q,
    k,
    v,
    g,
    do,
    dk,
    dv,
    dg,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
): ...
@triton.heuristics(
    {
        "NV": lambda args: triton.cdiv(args["V"], args["BV"]),
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config(triton_config, num_warps=num_warps) for num_warps in NUM_WARPS
    ],
    key=["BT", "BS", "BK", "BV", "USE_G"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def parallel_simple_gla_bwd_kernel(
    q,
    k,
    v,
    g,
    do,
    dq,
    dk,
    dv,
    dg,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
): ...
def parallel_simple_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    output_attentions: bool = ...,
    chunk_size: int = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Any, Tensor | Any, Tensor | None]: ...
def parallel_simple_gla_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    chunk_size: int = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Any, Any, Any, Tensor | Any | None]: ...

class ParallelSimpleGLAFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx, q, k, v, g, scale, output_attentions, cu_seqlens, cu_seqlens_cpu
    ) -> tuple[Any, Tensor | None]: ...
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx, do, da=...
    ) -> tuple[Any, Any, Any, Tensor | Any | None, None, None, None, None]: ...

def parallel_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = ...,
    scale: float | None = ...,
    output_attentions: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]: ...
