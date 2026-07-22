from fla.utils import autotune_cache_kwargs

import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h_init"] is not None,
        "STORE_FINAL_STATE": lambda args: args["h_final"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_mesa_net_fwd_kernel_h(
    k,
    v,
    beta,
    g,
    h,
    h_kv,
    h_init,
    h_kv_init,
    h_final,
    h_kv_final,
    cu_seqlens,
    split_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def chunk_mesa_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h_init: torch.Tensor,
    h_kv_init: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = ...,
    chunk_size: int = ...,
    split_size: int | None = ...,
    states_in_fp32: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
