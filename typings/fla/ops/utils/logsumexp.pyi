from fla.utils import autotune_cache_kwargs

import torch
import triton
import triton.language as tl

@triton.heuristics({"HAS_SCALE": lambda args: args["scale"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit
def logsumexp_fwd_kernel(
    x, z, scale, D: tl.constexpr, B: tl.constexpr, HAS_SCALE: tl.constexpr
): ...
def logsumexp_fwd(x, scale: float | None = ..., dtype: torch.dtype | None = ...): ...
