from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    input_guard,
)

import torch
import triton
import triton.language as tl

BKV_LIST = ...
NUM_WARPS = ...

@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_G_GAMMA": lambda args: args["g_gamma"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in BKV_LIST
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def fused_chunk_fwd_kernel(
    q,
    k,
    v,
    g,
    g_gamma,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_G_GAMMA": lambda args: args["g_gamma"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_INITIAL_STATE": lambda args: args["dh0"] is not None,
        "USE_FINAL_STATE": lambda args: args["dht"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def fused_chunk_bwd_kernel(
    q,
    k,
    v,
    g,
    g_gamma,
    do,
    dq,
    dk,
    dv,
    dg,
    h0,
    dht,
    dh0,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
): ...
def fused_chunk_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = ...,
    g_gamma: torch.Tensor | None = ...,
    scale: float | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
) -> tuple[Tensor | Any, Tensor | None]: ...
def fused_chunk_bwd(
    q,
    k,
    v,
    g,
    g_gamma,
    do,
    scale,
    initial_state: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
) -> tuple[Tensor | Any, Tensor | Any, Tensor | Any, Any | None, Tensor | None]: ...

class FusedChunkFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx, q, k, v, g, g_gamma, scale, initial_state, output_final_state, cu_seqlens
    ) -> tuple[Tensor | Any, Tensor | None]: ...
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx, do, dht=...
    ) -> tuple[
        Tensor | Any,
        Tensor | Any,
        Tensor | Any,
        Any | None,
        None,
        None,
        Tensor | None,
        None,
        None,
    ]: ...

def fused_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = ...,
    g_gamma: torch.Tensor | None = ...,
    scale: float | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
