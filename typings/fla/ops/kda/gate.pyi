from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    input_guard,
)

import torch
import triton
import triton.language as tl

BS_LIST = ...
BT_LIST_AUTOTUNE = ...
NUM_WARPS_AUTOTUNE = ...

def naive_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = ...,
    output_dtype: torch.dtype = ...,
) -> torch.Tensor: ...
def naive_kda_lowerbound_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = ...,
    lower_bound: float = ...,
    output_dtype: torch.dtype = ...,
) -> torch.Tensor: ...
@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "HAS_BETA": lambda args: args["beta"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in BT_LIST_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3]
    ],
    key=["H", "D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def kda_gate_fwd_kernel(
    g,
    A_log,
    dt_bias,
    beta,
    yg,
    yb,
    lower_bound,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_BETA: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
): ...
@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "HAS_BETA": lambda args: args["beta"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3]
    ],
    key=["H", "D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def kda_gate_bwd_kernel(
    g,
    A_log,
    dt_bias,
    beta,
    dyg,
    dyb,
    dg,
    dA,
    dbeta,
    lower_bound,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_BETA: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
): ...
def kda_gate_fwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = ...,
    lower_bound: float | None = ...,
    output_dtype: torch.dtype = ...,
) -> torch.Tensor: ...
def kda_gate_bwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = ...,
    dyg: torch.Tensor | None = ...,
    lower_bound: float | None = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...

class KDAGateFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        g: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor | None = ...,
        lower_bound: float | None = ...,
        output_dtype: torch.dtype = ...,
    ) -> torch.Tensor: ...
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx, dyg: torch.Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None, None, None]: ...

@torch.compiler.disable
def fused_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = ...,
    lower_bound: float | None = ...,
    output_dtype: torch.dtype = ...,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=["H", "S", "BT", "IS_VARLEN", "REVERSE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def kda_gate_chunk_cumsum_vector_kernel(
    s,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    lower_bound,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
): ...
@input_guard
def kda_gate_chunk_cumsum(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int,
    scale: float = ...,
    dt_bias: torch.Tensor | None = ...,
    cu_seqlens: torch.Tensor | None = ...,
    output_dtype: torch.dtype | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
    lower_bound: float | None = ...,
    **kwargs,
) -> torch.Tensor: ...
