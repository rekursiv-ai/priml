from typing import Any

from fla.utils import autotune_cache_kwargs, input_guard
from torch import Tensor, nn
from torch.distributed.tensor.parallel import ParallelStyle

import torch
import triton
import triton.language as tl

def layer_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = ...,
    eps: float = ...,
    prenorm: bool = ...,
    upcast: bool = ...,
) -> Tensor | tuple[Tensor, Tensor]: ...
def rms_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = ...,
    eps: float = ...,
    prenorm: bool = ...,
    upcast: bool = ...,
) -> Tensor | tuple[Tensor, Tensor]: ...
def group_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    residual: torch.Tensor = ...,
    eps: float = ...,
    is_rms_norm: bool = ...,
    prenorm: bool = ...,
    upcast: bool = ...,
) -> Tensor | tuple[Tensor, Tensor]: ...

class GroupNormRef(nn.Module):
    def __init__(
        self,
        num_groups: int,
        hidden_size: int,
        elementwise_affine: bool = ...,
        bias: bool = ...,
        eps: float = ...,
        is_rms_norm: bool = ...,
    ) -> GroupNormRef: ...
    def reset_parameters(self) -> None: ...
    def forward(
        self, x, residual=..., prenorm=...
    ) -> Tensor | tuple[Tensor, Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, Tensor]: ...

@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for BT in [32, 64, 128]
        for num_warps in [2, 4, 8]
    ],
    key=["D", "NB", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM"],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_fwd_kernel(
    x,
    y,
    w,
    b,
    res,
    res_out,
    mean,
    rstd,
    eps,
    T,
    G: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
): ...
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [2, 4, 8, 16]],
    key=["D", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM"],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_fwd_kernel1(
    x,
    y,
    w,
    b,
    res,
    res_out,
    mean,
    rstd,
    eps,
    G: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
): ...
@triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["y"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for BT in [32, 64]
        for num_warps in [2, 4, 8]
    ],
    key=["D", "NB", "HAS_DRESIDUAL", "STORE_DRESIDUAL", "IS_RMS_NORM"],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_bwd_kernel(
    x,
    w,
    b,
    y,
    dy,
    dx,
    dw,
    db,
    dres,
    dres_in,
    mean,
    rstd,
    T,
    G: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    GS: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
) -> None: ...
@triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["y"] is not None})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [2, 4, 8]],
    key=["D", "HAS_DRESIDUAL", "STORE_DRESIDUAL", "IS_RMS_NORM"],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_bwd_kernel1(
    x,
    w,
    b,
    y,
    dy,
    dx,
    dw,
    db,
    dres,
    dres_in,
    mean,
    rstd,
    T,
    G: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
    GS: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
) -> None: ...
def layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = ...,
    residual: torch.Tensor = ...,
    out_dtype: torch.dtype = ...,
    residual_dtype: torch.dtype = ...,
    is_rms_norm: bool = ...,
    num_groups: int = ...,
) -> tuple[Tensor, Tensor | None, Tensor, Tensor]: ...
def layer_norm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor = ...,
    rstd: torch.Tensor = ...,
    dres: torch.Tensor = ...,
    has_residual: bool = ...,
    is_rms_norm: bool = ...,
    x_dtype: torch.dtype = ...,
    recompute_output: bool = ...,
    num_groups: int = ...,
) -> (
    tuple[Tensor, Any | None, Any | None, Tensor | None]
    | tuple[Tensor, Any | None, Any | None, Tensor | None, Tensor | None]
): ...

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        weight,
        bias,
        residual: torch.Tensor = ...,
        eps: float = ...,
        prenorm: bool = ...,
        residual_in_fp32: bool = ...,
        is_rms_norm: bool = ...,
        num_groups: int = ...,
    ) -> Tensor | tuple[Tensor, Tensor]: ...
    @staticmethod
    @input_guard
    def backward(
        ctx, dy, *args
    ) -> tuple[
        Tensor, Any | None, Any | None, Tensor | None, None, None, None, None, None
    ]: ...

def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = ...,
    eps: float = ...,
    prenorm: bool = ...,
    residual_in_fp32: bool = ...,
    is_rms_norm: bool = ...,
) -> None: ...
def group_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = ...,
    eps: float = ...,
    prenorm: bool = ...,
    residual_in_fp32: bool = ...,
    is_rms_norm: bool = ...,
    num_groups: int = ...,
) -> None: ...
def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = ...,
    eps: float = ...,
    prenorm: bool = ...,
    residual_in_fp32: bool = ...,
) -> None: ...
def layer_norm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    residual: torch.Tensor = ...,
    eps: float = ...,
    prenorm: bool = ...,
    residual_in_fp32: bool = ...,
    is_rms_norm: bool = ...,
    num_groups: int = ...,
) -> None: ...
def rms_norm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    residual: torch.Tensor = ...,
    eps: float = ...,
    prenorm: bool = ...,
    residual_in_fp32: bool = ...,
) -> None: ...
def group_norm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    residual: torch.Tensor = ...,
    eps: float = ...,
    prenorm: bool = ...,
    residual_in_fp32: bool = ...,
    is_rms_norm: bool = ...,
    num_groups: int = ...,
) -> None: ...

class LayerNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = ...,
        bias: bool = ...,
        eps: float = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> LayerNorm: ...
    def reset_parameters(self) -> None: ...
    def forward(self, x, residual=..., prenorm=..., residual_in_fp32=...) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...

class GroupNorm(nn.Module):
    def __init__(
        self,
        num_groups: int,
        hidden_size: int,
        elementwise_affine: bool = ...,
        bias: bool = ...,
        eps: float = ...,
        is_rms_norm: bool = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> GroupNorm: ...
    def reset_parameters(self) -> None: ...
    def forward(self, x, residual=..., prenorm=..., residual_in_fp32=...) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...

class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = ...,
        bias: bool = ...,
        eps: float = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> RMSNorm: ...
    def reset_parameters(self) -> None: ...
    def forward(self, x, residual=..., prenorm=..., residual_in_fp32=...) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...

class LayerNormLinearFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual=...,
        eps=...,
        prenorm=...,
        residual_in_fp32=...,
        is_rms_norm=...,
        num_groups=...,
    ) -> tuple[Any, Tensor]: ...
    @staticmethod
    @input_guard
    def backward(
        ctx, dout, *args
    ) -> tuple[
        Tensor,
        Any | None,
        Any | None,
        Tensor,
        Any | None,
        Tensor | None,
        None,
        None,
        None,
        None,
        None,
    ]: ...

class LayerNormLinear(nn.Module):
    def __init__(
        self,
        hidden_size,
        elementwise_affine: bool = ...,
        bias: bool = ...,
        eps: float = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> LayerNormLinear: ...
    def reset_parameters(self) -> None: ...
    def forward(
        self, x, weight, bias, residual=..., prenorm=..., residual_in_fp32=...
    ) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...

class GroupNormLinear(nn.Module):
    def __init__(
        self,
        num_groups: int,
        hidden_size: int,
        elementwise_affine: bool = ...,
        bias: bool = ...,
        eps: float = ...,
        is_rms_norm: bool = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> GroupNormLinear: ...
    def reset_parameters(self) -> None: ...
    def forward(
        self, x, weight, bias, residual=..., prenorm=..., residual_in_fp32=...
    ) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...

class RMSNormLinear(nn.Module):
    def __init__(
        self,
        hidden_size,
        elementwise_affine: bool = ...,
        bias: bool = ...,
        eps: float = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> RMSNormLinear: ...
    def reset_parameters(self) -> None: ...
    def forward(
        self, x, weight, bias, residual=..., prenorm=..., residual_in_fp32=...
    ) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...

class NormParallel(ParallelStyle):
    def __init__(
        self, *, sequence_dim: int = ..., use_local_output: bool = ...
    ) -> None: ...
