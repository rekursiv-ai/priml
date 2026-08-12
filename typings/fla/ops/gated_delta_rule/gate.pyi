from torch import Tensor
from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    input_guard,
)

import torch
import triton
import triton.language as tl

def naive_gdn_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = ...,
    output_dtype: torch.dtype = ...,
) -> torch.Tensor: ...
@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["H", "BT", "IS_VARLEN", "REVERSE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def gdn_gate_chunk_cumsum_scalar_kernel(
    g,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics({"HAS_BIAS": lambda args: args["dt_bias"] is not None})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["H", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def gdn_gate_bwd_kernel(
    g,
    A_log,
    dt_bias,
    dyg,
    dg,
    dA,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
): ...
@input_guard
def gdn_gate_chunk_cumsum(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int,
    scale: float = ...,
    dt_bias: torch.Tensor | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
    output_dtype: torch.dtype | None = ...,
) -> torch.Tensor: ...
def gdn_gate_bwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    dyg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...
@triton.heuristics({"HAS_BIAS": lambda args: args["dt_bias"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in [32, 64, 128]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=["H"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def gdn_gate_fwd_kernel(
    g, A_log, dt_bias, yg, T, H: tl.constexpr, BT: tl.constexpr, HAS_BIAS: tl.constexpr
): ...
def gdn_gate_fwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = ...,
    output_dtype: torch.dtype = ...,
) -> torch.Tensor: ...

class GDNGateFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        g: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor | None = ...,
        output_dtype: torch.dtype = ...,
    ) -> torch.Tensor: ...
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx, dyg: torch.Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None, None]: ...

@torch.compiler.disable
def fused_gdn_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = ...,
    output_dtype: torch.dtype = ...,
) -> torch.Tensor: ...
