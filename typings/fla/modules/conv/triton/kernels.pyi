from fla.utils import autotune_cache_kwargs, input_guard

import torch
import triton
import triton.language as tl

NUM_WARPS_AUTOTUNE = ...
STATIC_WARPS = ...

@triton.heuristics(
    {
        "HAS_WEIGHT": lambda args: args["weight"] is not None,
        "HAS_BIAS": lambda args: args["bias"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BD": BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D", "W", "NB"],
    **autotune_cache_kwargs,
)
@triton.jit
def causal_conv1d_fwd_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics(
    {
        "HAS_WEIGHT": lambda args: args["dw"] is not None,
        "HAS_BIAS": lambda args: args["db"] is not None,
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "USE_FINAL_STATE": lambda args: args["dht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BD": BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in [4, 8, 16, 32]
    ],
    key=["D", "W", "NB"],
    **autotune_cache_kwargs,
)
@triton.jit
def causal_conv1d_bwd_kernel(
    x,
    y,
    weight,
    initial_state,
    dht,
    dy,
    dx,
    dw,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    stride_dx_n,
    stride_dx_t,
    stride_dx_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["cache"] is not None,
        "HAS_WEIGHT": lambda args: args["weight"] is not None,
        "HAS_BIAS": lambda args: args["bias"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BD": BD}, num_warps=num_warps)
        for BD in [8, 16, 32, 64, 128, 256]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D", "W"],
    restore_value=["cache"],
    **autotune_cache_kwargs,
)
@triton.jit
def causal_conv1d_update_kernel(
    x,
    cache,
    residual,
    y,
    weight,
    bias,
    stride_x_n,
    stride_x_d,
    stride_y_n,
    stride_y_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    BW: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
): ...
@triton.heuristics(
    {
        "USE_ACTIVATION": lambda args: args["y"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit
def compute_dh0_kernel(
    dy,
    y,
    weight,
    dh0,
    cu_seqlens,
    stride_dy_n,
    stride_dy_t,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
) -> None: ...
@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit
def causal_conv1d_states_fwd_kernel(
    x,
    initial_state,
    final_state,
    cu_seqlens,
    T,
    D,
    W,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    BD: tl.constexpr,
    BW: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
) -> None: ...
@input_guard(no_guard_contiguous=["x"])
def causal_conv1d_update_states(
    x: torch.Tensor,
    state_len: int,
    initial_state: torch.Tensor | None = ...,
    cu_seqlens: torch.Tensor | None = ...,
) -> torch.Tensor: ...
@input_guard(no_guard_contiguous=["x"])
def causal_conv1d_update(
    x: torch.Tensor,
    cache: torch.Tensor,
    residual: torch.Tensor | None = ...,
    weight: torch.Tensor | None = ...,
    bias: torch.Tensor | None = ...,
    activation: str | None = ...,
) -> torch.Tensor: ...
