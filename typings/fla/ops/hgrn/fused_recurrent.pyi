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
        triton.Config({"BD": BD}, num_warps=num_warps)
        for BD in [32, 64, 128]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_hgrn_fwd_kernel(
    x,
    g,
    o,
    h0,
    ht,
    cu_seqlens,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "USE_FINAL_STATE_GRADIENT": lambda args: args["dht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BD": BD}, num_warps=num_warps)
        for BD in [32, 64, 128]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_hgrn_bwd_kernel(
    g,
    o,
    h0,
    dx,
    dg,
    do,
    dht,
    dh0,
    cu_seqlens,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def fused_recurrent_hgrn_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor = ...,
    output_final_state: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def fused_recurrent_hgrn_bwd(
    g: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor = ...,
    initial_state: torch.Tensor = ...,
    cu_seqlens: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...

class FusedRecurrentHGRNFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        g: torch.Tensor,
        initial_state: torch.Tensor = ...,
        output_final_state: bool = ...,
        cu_seqlens: torch.LongTensor | None = ...,
    ) -> tuple[Tensor, Tensor]: ...
    @staticmethod
    @input_guard
    def backward(ctx, do, dht=...) -> tuple[Tensor, Tensor, Any, None, None]: ...

@torch.compiler.disable
def fused_recurrent_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor = ...,
    output_final_state: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
