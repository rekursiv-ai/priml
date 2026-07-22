from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    input_guard,
)

import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [4, 8]],
    key=["BK", "BV", "USE_G", "USE_G_GAMMA", "USE_GK", "USE_GV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["B", "T"])
def fused_recurrent_fwd_kernel(
    q,
    k,
    v,
    g,
    g_gamma,
    gk,
    gv,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_INITIAL_STATE_GRADIENT": lambda args: args["dh0"] is not None,
        "USE_FINAL_STATE_GRADIENT": lambda args: args["dht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [4]],
    key=["BK", "BV", "USE_G", "USE_G_GAMMA", "USE_GK", "USE_GV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["B", "T"])
def fused_recurrent_bwd_kernel(
    q,
    k,
    v,
    g,
    g_gamma,
    gk,
    gv,
    o,
    h0,
    do,
    dq,
    dk,
    dv,
    dg,
    dgk,
    dgv,
    dht,
    dh0,
    cu_seqlens,
    scale,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def fused_recurrent_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = ...,
    g_gamma: torch.Tensor | None = ...,
    gk: torch.Tensor | None = ...,
    gv: torch.Tensor | None = ...,
    scale: float | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    reverse: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
) -> tuple[Any, Tensor | None]: ...
def fused_recurrent_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = ...,
    g_gamma: torch.Tensor | None = ...,
    gk: torch.Tensor | None = ...,
    gv: torch.Tensor | None = ...,
    o: torch.Tensor | None = ...,
    do: torch.Tensor | None = ...,
    dht: torch.Tensor | None = ...,
    scale: float | None = ...,
    initial_state: torch.Tensor | None = ...,
    reverse: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
) -> tuple[Any, Any, Any, Any | None, Any | None, Any | None, Tensor | None]: ...

class FusedRecurrentFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = ...,
        g_gamma: torch.Tensor | None = ...,
        gk: torch.Tensor | None = ...,
        gv: torch.Tensor | None = ...,
        scale: float | None = ...,
        initial_state: torch.Tensor | None = ...,
        output_final_state: bool = ...,
        reverse: bool = ...,
        cu_seqlens: torch.LongTensor | None = ...,
    ) -> tuple[Any, Tensor | None]: ...
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx, do, dht
    ) -> tuple[
        Any,
        Any,
        Any,
        Any | None,
        None,
        Any | None,
        Any | None,
        None,
        Tensor | None,
        None,
        None,
        None,
    ]: ...

def fused_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = ...,
    g_gamma: torch.Tensor | None = ...,
    gk: torch.Tensor | None = ...,
    gv: torch.Tensor | None = ...,
    scale: float | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    reverse: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
) -> Any | None: ...
