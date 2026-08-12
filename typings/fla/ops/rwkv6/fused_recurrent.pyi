from torch import Tensor
from typing import Any
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
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16]],
    key=["BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_rwkv6_fwd_kernel(
    q,
    k,
    v,
    w,
    u,
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
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=["BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_rwkv6_bwd_kernel_dq(
    k,
    v,
    w,
    u,
    do,
    dq,
    dq1,
    h0,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["dh0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=["BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_rwkv6_bwd_kernel_dkv(
    q,
    k,
    v,
    w,
    u,
    do,
    dk,
    dk1,
    dv,
    dh0,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BT": BT, "BK": BK}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for BK in [32, 64]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["K"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_rwkv6_bwd_kernel_dw(
    q,
    k,
    dq,
    dk,
    dw,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    REVERSE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def fused_recurrent_rwkv6_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: float | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    reverse: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
) -> tuple[Any, Tensor | None]: ...
def fused_recurrent_rwkv6_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    do: torch.Tensor,
    scale: float | None = ...,
    initial_state: torch.Tensor | None = ...,
    reverse: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
) -> tuple[Any, Any, Any, Tensor, Tensor | Any, Tensor | None]: ...

class FusedRecurrentRWKV6Function(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
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
        Any, Any, Any, Tensor, Tensor | Any, None, Tensor | None, None, None, None
    ]: ...

def fused_recurrent_rwkv6(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: float | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    reverse: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]: ...
