from typing import Any

from fla.utils import input_guard
from torch import Tensor, nn

import torch
import triton
import triton.language as tl

def rms_norm_ref(
    x, weight, bias, z=..., eps=..., group_size=..., norm_before_gate=..., upcast=...
): ...
@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["B"] is not None,
        "HAS_Z": lambda args: args["Z"] is not None,
    }
)
@triton.jit
def layer_norm_fwd_kernel(
    X,
    Y,
    W,
    B,
    Z,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
): ...
def layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    z: torch.Tensor = ...,
    out: torch.Tensor = ...,
    group_size: int = ...,
    norm_before_gate: bool = ...,
    is_rms_norm: bool = ...,
) -> tuple[Tensor, Tensor | None, Tensor]: ...
@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["B"] is not None,
        "HAS_Z": lambda args: args["Z"] is not None,
        "RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None,
    }
)
@triton.jit
def layer_norm_bwd_kernel(
    X,
    W,
    B,
    Z,
    Y,
    DY,
    DX,
    DW,
    DB,
    DZ,
    Mean,
    Rstd,
    stride_x_row,
    stride_z_row,
    stride_y_row,
    stride_dy_row,
    stride_dx_row,
    stride_dz_row,
    stride_dw_row,
    stride_db_row,
    M,
    N,
    eps,
    rows_per_program,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
): ...
def layer_norm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    z: torch.Tensor = ...,
    group_size: int = ...,
    norm_before_gate: bool = ...,
    is_rms_norm: bool = ...,
    recompute_output: bool = ...,
    dz: torch.Tensor = ...,
    out: torch.Tensor = ...,
) -> (
    tuple[Tensor, Tensor, Tensor | None, Tensor]
    | tuple[Tensor, Tensor, Tensor | None, Tensor, Tensor]
): ...

class LayerNormFn(torch.autograd.Function):
    @input_guard
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        z=...,
        eps=...,
        group_size=...,
        norm_before_gate=...,
        is_rms_norm=...,
    ) -> Tensor: ...
    @input_guard
    @staticmethod
    def backward(
        ctx, dy
    ) -> tuple[
        Tensor, Tensor, Tensor | None, Tensor | None, None, None, None, None
    ]: ...

def layernorm_fn(
    x,
    weight,
    bias,
    z=...,
    eps=...,
    group_size=...,
    norm_before_gate=...,
    is_rms_norm=...,
) -> None: ...
def rmsnorm_fn(
    x, weight, bias, z=..., eps=..., group_size=..., norm_before_gate=...
) -> None: ...

class LayerNormGated(nn.Module):
    def __init__(
        self,
        hidden_size,
        eps: float = ...,
        group_size: int | None = ...,
        norm_before_gate: bool = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, x, z=...) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...

class RMSNormGated(nn.Module):
    def __init__(
        self,
        hidden_size,
        eps: float = ...,
        group_size: int | None = ...,
        norm_before_gate: bool = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, x, z=...) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...
