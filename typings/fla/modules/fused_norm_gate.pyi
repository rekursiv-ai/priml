from fla.utils import autotune_cache_kwargs, input_guard
from torch import nn

import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "STORE_RESIDUAL_OUT": lambda args: args["residual_out"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "HAS_WEIGHT": lambda args: args["w"] is not None,
        "HAS_BIAS": lambda args: args["b"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for num_warps in [4, 8, 16]
    ],
    key=["D", "NB", "IS_RMS_NORM", "STORE_RESIDUAL_OUT", "HAS_RESIDUAL", "HAS_WEIGHT"],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_gated_fwd_kernel(
    x,
    g,
    y,
    w,
    b,
    residual,
    residual_out,
    mean,
    rstd,
    eps,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
): ...
@triton.heuristics(
    {
        "STORE_RESIDUAL_OUT": lambda args: args["residual_out"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "HAS_WEIGHT": lambda args: args["w"] is not None,
        "HAS_BIAS": lambda args: args["b"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [2, 4, 8, 16]],
    key=["D", "IS_RMS_NORM", "STORE_RESIDUAL_OUT", "HAS_RESIDUAL", "HAS_WEIGHT"],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_gated_fwd_kernel1(
    x,
    g,
    y,
    w,
    b,
    residual,
    residual_out,
    mean,
    rstd,
    eps,
    D: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
): ...
@triton.heuristics(
    {
        "HAS_DRESIDUAL": lambda args: args["dresidual"] is not None,
        "HAS_WEIGHT": lambda args: args["w"] is not None,
        "HAS_BIAS": lambda args: args["b"] is not None,
        "RECOMPUTE_OUTPUT": lambda args: args["y"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for num_warps in [4, 8, 16]
    ],
    key=["D", "NB", "IS_RMS_NORM", "HAS_DRESIDUAL", "HAS_WEIGHT"],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_gated_bwd_kernel(
    x,
    g,
    w,
    b,
    y,
    dy,
    dx,
    dg,
    dw,
    db,
    dresidual,
    dresidual_in,
    mean,
    rstd,
    T,
    BS,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
) -> None: ...
@triton.heuristics(
    {
        "HAS_DRESIDUAL": lambda args: args["dresidual"] is not None,
        "HAS_WEIGHT": lambda args: args["w"] is not None,
        "HAS_BIAS": lambda args: args["b"] is not None,
        "RECOMPUTE_OUTPUT": lambda args: args["y"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [2, 4, 8, 16]],
    key=["D", "IS_RMS_NORM", "STORE_DRESIDUAL", "HAS_DRESIDUAL", "HAS_WEIGHT"],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_gated_bwd_kernel1(
    x,
    g,
    w,
    b,
    y,
    dy,
    dx,
    dg,
    dw,
    db,
    dresidual,
    dresidual_in,
    mean,
    rstd,
    T,
    BS,
    D: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
) -> None: ...
def layer_norm_gated_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = ...,
    eps: float = ...,
    residual: torch.Tensor = ...,
    out_dtype: torch.dtype = ...,
    residual_dtype: torch.dtype = ...,
    is_rms_norm: bool = ...,
) -> tuple[Tensor, Tensor | None, Tensor, Tensor]: ...
def layer_norm_gated_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = ...,
    eps: float = ...,
    mean: torch.Tensor = ...,
    rstd: torch.Tensor = ...,
    dresidual: torch.Tensor = ...,
    has_residual: bool = ...,
    is_rms_norm: bool = ...,
    x_dtype: torch.dtype = ...,
    recompute_output: bool = ...,
) -> (
    tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]
    | tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]
): ...

class LayerNormGatedFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        activation: str,
        residual: torch.Tensor | None = ...,
        eps: float = ...,
        prenorm: bool = ...,
        residual_in_fp32: bool = ...,
        is_rms_norm: bool = ...,
    ) -> Tensor | tuple[Tensor, Tensor]: ...
    @staticmethod
    @input_guard
    def backward(
        ctx, dy, *args
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor | None,
        Tensor | None,
        None,
        Tensor | None,
        None,
        None,
        None,
        None,
    ]: ...

class LayerNormGatedLinearFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        g: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        linear_weight: torch.Tensor,
        linear_bias: torch.Tensor,
        residual: torch.Tensor | None = ...,
        eps: float = ...,
        prenorm: bool = ...,
        residual_in_fp32: bool = ...,
        is_rms_norm: bool = ...,
    ) -> tuple[Any, Tensor]: ...
    @staticmethod
    @input_guard
    def backward(
        ctx, dout, *args
    ) -> tuple[
        Tensor,
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

def layer_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = ...,
    residual: torch.Tensor | None = ...,
    prenorm: bool = ...,
    residual_in_fp32: bool = ...,
    eps: float = ...,
) -> Any | None: ...
def rms_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = ...,
    residual: torch.Tensor | None = ...,
    prenorm: bool = ...,
    residual_in_fp32: bool = ...,
    eps: float = ...,
) -> Any | None: ...
def layer_norm_swish_gate_linear(
    x: torch.Tensor,
    g: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    residual: torch.Tensor | None = ...,
    prenorm: bool = ...,
    residual_in_fp32: bool = ...,
    eps: float = ...,
) -> Any | None: ...
def rms_norm_swish_gate_linear(
    x,
    g: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    residual: torch.Tensor | None = ...,
    prenorm: bool = ...,
    residual_in_fp32: bool = ...,
    eps: float = ...,
) -> Any | None: ...

class FusedLayerNormGated(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = ...,
        bias: bool = ...,
        activation: str = ...,
        eps: float = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> FusedLayerNormGated: ...
    def reset_parameters(self) -> None: ...
    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = ...,
        prenorm: bool = ...,
        residual_in_fp32: bool = ...,
    ) -> torch.Tensor: ...

class FusedRMSNormGated(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = ...,
        eps: float = ...,
        activation: str = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> FusedRMSNormGated: ...
    def reset_parameters(self) -> None: ...
    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = ...,
        prenorm: bool = ...,
        residual_in_fp32: bool = ...,
    ) -> torch.Tensor: ...

class FusedLayerNormSwishGate(FusedLayerNormGated):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = ...,
        bias: bool = ...,
        eps: float = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> FusedLayerNormSwishGate: ...

class FusedRMSNormSwishGate(FusedRMSNormGated):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = ...,
        eps: float = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> FusedRMSNormSwishGate: ...

class FusedLayerNormGatedLinear(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = ...,
        eps: float = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> FusedLayerNormGatedLinear: ...
    def reset_parameters(self) -> None: ...
    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor | None = ...,
        bias: torch.Tensor | None = ...,
        residual: torch.Tensor | None = ...,
        prenorm: bool = ...,
        residual_in_fp32: bool = ...,
    ) -> torch.Tensor: ...

class FusedLayerNormSwishGateLinear(FusedLayerNormGatedLinear):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = ...,
        eps: float = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> FusedLayerNormSwishGateLinear: ...

class FusedRMSNormGatedLinear(nn.Module):
    def __init__(
        self,
        hidden_size,
        elementwise_affine: bool = ...,
        eps: float = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> FusedRMSNormGatedLinear: ...
    def reset_parameters(self) -> None: ...
    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor | None = ...,
        bias: torch.Tensor | None = ...,
        residual: torch.Tensor | None = ...,
        prenorm: bool = ...,
        residual_in_fp32: bool = ...,
    ) -> torch.Tensor: ...

class FusedRMSNormSwishGateLinear(FusedRMSNormGatedLinear):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = ...,
        eps: float = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> FusedRMSNormSwishGateLinear: ...
