from typing import Any

from fla.utils import autotune_cache_kwargs, input_guard, require_version
from torch import Tensor, nn

import torch
import triton
import triton.language as tl

NUM_WARPS_AUTOTUNE = ...

def activation_quant(x): ...
def weight_quant(w): ...
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS"],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_fwd_kernel_quant(
    X,
    Y,
    W,
    B,
    RESIDUAL,
    RESIDUAL_OUT,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_res_row,
    stride_res_out_row,
    N,
    eps,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
): ...
def layer_norm_fwd_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    residual: torch.Tensor = ...,
    out_dtype: torch.dtype = ...,
    residual_dtype: torch.dtype = ...,
    is_rms_norm: bool = ...,
) -> tuple[Tensor, Tensor | None, Tensor, Tensor]: ...
@triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["N", "HAS_DRESIDUAL", "STORE_DRESIDUAL", "IS_RMS_NORM", "HAS_BIAS"],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_bwd_kernel(
    X,
    W,
    B,
    Y,
    DY,
    DX,
    DW,
    DB,
    DRESIDUAL,
    DRESIDUAL_IN,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_dy_row,
    stride_dx_row,
    stride_dres_row,
    stride_dres_in_row,
    M,
    N,
    eps,
    rows_per_program,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
) -> None: ...
def layer_norm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    dresidual: torch.Tensor = ...,
    has_residual: bool = ...,
    is_rms_norm: bool = ...,
    x_dtype: torch.dtype = ...,
    recompute_output: bool = ...,
) -> (
    tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]
    | tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]
): ...

class LayerNormLinearQuantFn(torch.autograd.Function):
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
    ) -> tuple[Any, Tensor]: ...
    @staticmethod
    @input_guard
    def backward(
        ctx, dout, *args
    ) -> tuple[
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor,
        Any | None,
        Tensor | None,
        None,
        None,
        None,
        None,
    ]: ...

def layer_norm_linear_quant_fn(
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
) -> None: ...
def rms_norm_linear_quant(
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
@require_version("triton>=3.0", ...)
def bit_linear(
    x, weight, bias=..., norm_weight=..., norm_bias=..., eps=...
) -> None: ...

class BitLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = ...,
        norm_eps: float = ...,
    ) -> None: ...
    def forward(self, x): ...

class FusedBitLinear(BitLinear):
    def __init__(self, in_features, out_features, bias=...) -> None: ...
    def forward(self, x) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...
