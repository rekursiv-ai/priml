from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["T"])
def parallel_based_fwd_kernel(
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
def parallel_based_bwd_kernel(
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

triton_parallel_based = ...

def parallel_based(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = ...,
    use_norm: bool = ...,
    head_first: bool = ...,
) -> Any: ...
