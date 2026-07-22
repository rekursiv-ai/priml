from fla.utils import autotune_cache_kwargs, input_guard

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
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [32, 64]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=["BK"],
    **autotune_cache_kwargs,
)
@triton.jit
def fused_recurrent_fwd_kernel(
    q,
    k,
    v,
    a,
    b,
    o,
    ha,
    h0,
    ht,
    cu_seqlens,
    scale,
    H,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "USE_DHT": lambda args: args["dht"] is not None,
        "USE_DH0": lambda args: args["dh0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3]
    ],
    key=["BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit
def fused_recurrent_bwd_kernel(
    q,
    k,
    v,
    a,
    b,
    ha,
    dht,
    dh0,
    do,
    dq,
    dk,
    dv,
    da,
    db,
    dha,
    h0,
    scale,
    cu_seqlens,
    B,
    H,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_DH0: tl.constexpr,
    USE_DHT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...

class FusedRecurrentIPLRDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        scale: float | None = ...,
        initial_state: torch.Tensor | None = ...,
        output_final_state: bool = ...,
        cu_seqlens: torch.LongTensor | None = ...,
    ) -> tuple[Tensor, Tensor | None]: ...
    @staticmethod
    @input_guard
    def backward(
        ctx, do, dht
    ) -> tuple[Any, Any, Tensor, Any, Any, None, Tensor | None, None, None]: ...

def fused_recurrent_iplr_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = ...,
    initial_state: torch.Tensor = ...,
    output_final_state: bool = ...,
    cu_seqlens: torch.Tensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
