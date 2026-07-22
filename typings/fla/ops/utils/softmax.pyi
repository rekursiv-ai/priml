from fla.utils import autotune_cache_kwargs

import torch
import triton
import triton.language as tl

NUM_WARPS_AUTOTUNE = ...

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit
def softmax_fwd_kernel(x, p, D: tl.constexpr, B: tl.constexpr): ...
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit
def softmax_bwd_kernel(p, dp, ds, D: tl.constexpr, B: tl.constexpr): ...
def softmax_fwd(x: torch.Tensor, dtype: torch.dtype | None = ...) -> torch.Tensor: ...
def softmax_bwd(
    p: torch.Tensor, dp: torch.Tensor, dtype: torch.dtype | None = ...
) -> torch.Tensor: ...
