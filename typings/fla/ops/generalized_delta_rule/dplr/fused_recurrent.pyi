from fla.utils import (
    USE_CUDA_GRAPH,
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
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [16, 32, 64]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=["BK"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_dplr_delta_rule_fwd_kernel(
    q,
    k,
    v,
    a,
    b,
    gk,
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
def fused_recurrent_dplr_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    reverse: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
) -> tuple[Tensor, Tensor | None]: ...

class FusedRecurrentDPLRDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        gk: torch.Tensor,
        scale: float | None = ...,
        initial_state: torch.Tensor | None = ...,
        output_final_state: bool = ...,
        reverse: bool = ...,
        cu_seqlens: torch.LongTensor | None = ...,
    ) -> tuple[Tensor, Tensor | None]: ...
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht): ...

def fused_recurrent_dplr_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    reverse: bool = ...,
    cu_seqlens: torch.Tensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
