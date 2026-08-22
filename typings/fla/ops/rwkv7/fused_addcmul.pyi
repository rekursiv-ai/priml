from typing import Any

from fla.utils import (
    USE_CUDA_GRAPH,
    autotune_cache_kwargs,
    input_guard,
)
from torch import Tensor

import torch
import triton
import triton.language as tl

logger = ...

def identity_decorator(fn): ...

current_python_version = ...
min_torch_compile_version = ...
fla_use_compile = ...
if current_python_version >= min_torch_compile_version and fla_use_compile:
    torch_compile = ...
else:
    torch_compile = ...
NUM_WARPS_AUTOTUNE = ...

@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [1, 2, 3]
        for BT in [2, 4, 8]
    ],
    key=["BD"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit
def fused_addcmul_fwd_kernel(
    hidden,
    delta,
    ixr,
    ixw,
    ixk,
    ixv,
    ixa,
    ixg,
    oxr,
    oxw,
    oxk,
    oxv,
    oxa,
    oxg,
    use_xg: tl.constexpr,
    T,
    T_OFFSET,
    BT: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
) -> None: ...
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [1, 2, 3]
        for BT in [2, 4, 8]
    ],
    key=["BD"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit
def addcmul_bwd_kernel1(
    ixr,
    ixw,
    ixk,
    ixv,
    ixa,
    ixg,
    dxr,
    dxw,
    dxk,
    dxv,
    dxa,
    dxg,
    ghidden,
    gx,
    use_xg: tl.constexpr,
    T,
    T_OFFSET,
    BT: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    DTYPE: tl.constexpr,
) -> None: ...
def addcmul_bwd1(
    d_xr,
    d_xw,
    d_xk,
    d_xv,
    d_xa,
    d_xg,
    x_r,
    x_w,
    x_k,
    x_v,
    x_a,
    x_g,
    hidden_states,
    delta,
    use_xg,
    inplace=...,
) -> tuple[Any | Tensor, Tensor]: ...
@torch_compile
def addcmul_bwd2(
    d_oxr, d_xw, d_xk, d_xv, d_xa, d_xg, delta, use_xg: bool
) -> tuple[Any, Any, Any, Any, Any, Any | None]: ...

class Rwkv7FusedAddcmul(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx, hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]: ...
    @staticmethod
    @input_guard
    def backward(
        ctx, dxr, dxw, dxk, dxv, dxa, dxg
    ) -> tuple[Any | Tensor, Tensor, Any, Any, Any, Any, Any, Any | None]: ...

def fused_addcmul_rwkv7(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    xr: torch.Tensor,
    xw: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    xa: torch.Tensor,
    xg: torch.Tensor | None = ...,
) -> (
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor, Tensor, Tensor, None]
    | Any
    | None
): ...
def torch_addcmul_rwkv7(
    hidden_states, delta, xr, xw, xk, xv, xa, xg=...
) -> (
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor, Tensor, Tensor, None]
): ...
