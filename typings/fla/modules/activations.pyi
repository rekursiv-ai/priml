from torch import Tensor
from typing import Any
from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    input_guard,
)

import torch
import triton
import triton.language as tl

NUM_WARPS_AUTOTUNE = ...

@triton.autotune(
    configs=[
        triton.Config({"B": bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def sigmoid_fwd_kernel(
    x, y, T, D: tl.constexpr, stride_x_row, stride_y_row, B: tl.constexpr
): ...
@triton.autotune(
    configs=[
        triton.Config({"B": bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def sigmoid_bwd_kernel(
    x,
    dy,
    dx,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_dy_row,
    stride_dx_row,
    B: tl.constexpr,
): ...
@torch.compiler.disable
def sigmoid_fwd(x: torch.Tensor, output_contiguous: bool = ...) -> torch.Tensor: ...
@torch.compiler.disable
def sigmoid_bwd(
    x: torch.Tensor, dy: torch.Tensor, output_contiguous: bool = ...
) -> torch.Tensor: ...

class SigmoidFunction(torch.autograd.Function):
    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def forward(ctx, x) -> Tensor: ...
    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def backward(ctx, dout) -> Tensor: ...

sigmoid = ...

@triton.autotune(
    configs=[
        triton.Config({"B": bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def logsigmoid_fwd_kernel(
    x, y, temperature, T, D: tl.constexpr, stride_x_row, stride_y_row, B: tl.constexpr
): ...
@triton.autotune(
    configs=[
        triton.Config({"B": bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def logsigmoid_bwd_kernel(
    x,
    dx,
    dy,
    temperature,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_dx_row,
    stride_dy_row,
    B: tl.constexpr,
): ...
@torch.compiler.disable
def logsigmoid_fwd(
    x: torch.Tensor, temperature: float = ..., output_contiguous: bool = ...
) -> torch.Tensor: ...
@torch.compiler.disable
def logsigmoid_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    temperature: float = ...,
    output_contiguous: bool = ...,
) -> torch.Tensor: ...

class LogSigmoidFunction(torch.autograd.Function):
    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def forward(ctx, x, temperature) -> Tensor: ...
    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def backward(ctx, dy) -> tuple[Tensor, None]: ...

def logsigmoid(x: torch.Tensor, temperature: float = ...) -> torch.Tensor: ...
@triton.autotune(
    configs=[
        triton.Config({"B": bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def swish_fwd_kernel(
    x, y, T, D: tl.constexpr, stride_x_row, stride_y_row, B: tl.constexpr
): ...
@triton.autotune(
    configs=[
        triton.Config({"B": bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def swish_bwd_kernel(
    x,
    dy,
    dx,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_dy_row,
    stride_dx_row,
    B: tl.constexpr,
): ...
@torch.compiler.disable
def swish_fwd(x: torch.Tensor, output_contiguous: bool = ...) -> torch.Tensor: ...
@torch.compiler.disable
def swish_bwd(
    x: torch.Tensor, dy: torch.Tensor, output_contiguous: bool = ...
) -> torch.Tensor: ...

class SwishFunction(torch.autograd.Function):
    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def forward(ctx, x) -> Tensor: ...
    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def backward(ctx, dout) -> Tensor: ...

swish = ...

@torch.compile
def bias_gelu(y, bias): ...
@torch.compile
def bias_gelu_bwd(g, y, bias) -> tuple[Any, Any]: ...

class GeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias): ...
    @staticmethod
    def backward(ctx, grad_output) -> tuple[tuple[Any, Any], tuple[Any, Any]]: ...

bias_gelu_impl = ...

@torch.compile
def gelu_fwd(x): ...
@torch.compile
def gelu_bwd(g, x): ...

class FastGeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input): ...
    @staticmethod
    def backward(ctx, grad_output): ...

fast_gelu_impl = ...

@torch.compile
def relu_bwd(g, x) -> Tensor: ...
@torch.compile
def sqrelu_fwd(x) -> Tensor: ...
@torch.compile
def sqrelu_bwd(g, x): ...

class SquaredReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input) -> Tensor: ...
    @staticmethod
    def backward(ctx, grad_output): ...

sqrelu = ...

@triton.autotune(
    configs=[
        triton.Config({"B": bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def swiglu_fwd_kernel(
    x,
    y,
    z,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    B: tl.constexpr,
): ...
@triton.heuristics({"HAS_WEIGHT": lambda args: args["z"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"B": bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def swiglu_fwdbwd_kernel(
    x,
    y,
    g,
    dx,
    dy,
    z,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_y_row,
    stride_g_row,
    stride_dx_row,
    stride_dy_row,
    stride_z_row,
    B: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
): ...
@torch.compiler.disable
def swiglu_fwd(
    x: torch.Tensor, y: torch.Tensor, output_contiguous: bool = ...
) -> torch.Tensor: ...
@torch.compiler.disable
def swiglu_fwdbwd(
    x: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    use_weight: bool = ...,
    output_contiguous: bool = ...,
) -> tuple[Tensor, Tensor, Tensor | None] | tuple[Tensor, Tensor]: ...

class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def forward(ctx, x, y) -> Tensor: ...
    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def backward(
        ctx, dout
    ) -> tuple[Tensor, Tensor, Tensor | None] | tuple[Tensor, Tensor]: ...

class SwiGLULinearFunction(torch.autograd.Function):
    @staticmethod
    @input_guard(no_guard_contiguous=True)
    @autocast_custom_fwd
    def forward(ctx, x, y, weight, bias): ...
    @staticmethod
    @input_guard(no_guard_contiguous=True)
    @autocast_custom_bwd
    def backward(ctx, dout, *args) -> tuple[Tensor, Tensor, Tensor, Any | None]: ...

swiglu = ...
swiglu_linear = ...
ACT2FN = ...
