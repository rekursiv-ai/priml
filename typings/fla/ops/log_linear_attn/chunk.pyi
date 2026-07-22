from dataclasses import dataclass

from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    input_guard,
)

import torch
import triton
import triton.language as tl

BLOCK_K = ...

@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BLOCK_K}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_fwd_kernel(
    q,
    k,
    v,
    g,
    level_scales,
    llut,
    o,
    h0,
    ht,
    offsets,
    new_offsets,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    L_IN: tl.constexpr,
    L_OUT: tl.constexpr,
    MIN_LEVEL: tl.constexpr,
    MAX_LEVEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
) -> None: ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def copy_input_kernel(
    q,
    k,
    v,
    g,
    level_scales,
    cu_seqlens,
    q_prev,
    k_prev,
    v_prev,
    g_prev,
    level_scales_prev,
    offsets,
    q_new,
    k_new,
    v_new,
    g_new,
    level_scales_new,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def copy_last_chunk_kernel(
    q,
    k,
    v,
    g,
    level_scales,
    cu_seqlens,
    q_prev,
    k_prev,
    v_prev,
    g_prev,
    level_scales_prev,
    offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
) -> None: ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V"],
    restore_value=["dh", "dg_last"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_bwd_kernel_dhg(
    do,
    q,
    g,
    l,
    h_l,
    dh,
    dg_last,
    ell,
    T,
    cu_seqlens,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    NT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V"],
    restore_value=["dq", "dg"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_bwd_kernel_hdqgl(
    do,
    q,
    k,
    v,
    g,
    l,
    h_l,
    dq,
    dg,
    dl,
    ell,
    T,
    cu_seqlens,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V"],
    restore_value=["dk", "dg", "dg_last"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_bwd_kernel_dkg(
    dh,
    k,
    v,
    g,
    dg_last,
    dk,
    dg,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V"],
    restore_value=["dv"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_bwd_kernel_dv(
    dh,
    k,
    g,
    dv,
    T,
    cu_seqlens,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V"],
    restore_value=["dl", "dq", "dk", "dv", "dg"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_bwd_kernel_diag(
    do,
    q,
    k,
    v,
    g,
    l,
    llut,
    mask,
    dq,
    dk,
    dv,
    dg,
    dl,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def construct_binary_level_mask(level, T) -> Tensor: ...
def level_lut(BT, device) -> Tensor: ...
def masks(BT, device) -> Tensor: ...
def ceil_div(x: int, y: int) -> int: ...
def ceil_log(x: int, b: int) -> int: ...

@dataclass
class LogLinearAttentionState:
    ht: torch.Tensor
    offsets: torch.Tensor
    q_prev: torch.Tensor
    k_prev: torch.Tensor
    v_prev: torch.Tensor
    g_prev: torch.Tensor
    level_scales_prev: torch.Tensor

class ChunkLogLinearAttentionFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx, q, k, v, g, level_scales, initial_state, output_final_state, cu_seqlens
    ) -> tuple[Tensor, LogLinearAttentionState] | tuple[Tensor, None]: ...
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx, do, dht
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | Any, Tensor, None, None, None]: ...

@torch.compiler.disable
def chunk_log_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    level_scales: torch.Tensor,
    initial_state: LogLinearAttentionState | None = ...,
    output_final_state: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
