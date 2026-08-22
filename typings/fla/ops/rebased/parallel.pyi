from typing import Any

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from torch import Tensor

import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["T"])
def parallel_rebased_fwd_kernel(
    q,
    k,
    v,
    o,
    z,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
): ...
@triton.jit(do_not_specialize=["T"])
def parallel_rebased_bwd_kernel(
    q,
    k,
    v,
    do,
    dz,
    dq,
    dk,
    dv,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
): ...

class ParallelBasedFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, scale) -> tuple[Tensor, Tensor]: ...
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dz) -> tuple[Tensor, Tensor, Tensor, None]: ...

def parallel_rebased(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = ...,
    use_scale: bool = ...,
    use_normalize: bool = ...,
    return_both: bool = ...,
    head_first: bool = ...,
) -> tuple[Any, Any] | Any: ...
