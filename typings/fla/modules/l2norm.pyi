from typing import Any

from fla.utils import autotune_cache_kwargs, input_guard
from torch import Tensor, nn

import torch
import triton
import triton.language as tl

BT_LIST = ...
NUM_WARPS_AUTOTUNE = ...

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit
def l2norm_fwd_kernel1(x, y, rstd, eps, D, BD: tl.constexpr): ...
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit
def l2norm_bwd_kernel1(y, rstd, dy, dx, eps, D, BD: tl.constexpr): ...
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
        for BT in BT_LIST
    ],
    key=["D", "NB"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def l2norm_fwd_kernel(
    x,
    y,
    rstd,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    BT: tl.constexpr,
): ...
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
        for BT in BT_LIST
    ],
    key=["D", "NB"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def l2norm_bwd_kernel(
    y,
    rstd,
    dy,
    dx,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    BT: tl.constexpr,
): ...
def l2norm_fwd(
    x: torch.Tensor, eps: float = ..., output_dtype: torch.dtype | None = ...
) -> tuple[Tensor, Tensor]: ...
def l2norm_bwd(
    y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor, eps: float = ...
) -> Tensor: ...

class L2NormFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(ctx, x, eps=..., output_dtype=...) -> Tensor: ...
    @staticmethod
    @input_guard
    def backward(ctx, dy) -> tuple[Tensor, None, None]: ...

def l2norm(
    x: torch.Tensor, eps: float = ..., output_dtype: torch.dtype | None = ...
) -> torch.Tensor: ...

l2_norm = ...

class L2Norm(nn.Module):
    def __init__(
        self, eps: float = ..., output_dtype: torch.dtype | None = ...
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
