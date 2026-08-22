from typing import Any

from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    input_guard,
)
from torch import Tensor

import torch
import triton
import triton.language as tl

def gate_output_correction_ref(
    o: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    r_k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
) -> Tensor: ...
def gate_output_correction_backward_ref(
    grad_output, o, r, k, r_k, v, g
) -> tuple[Any, Any, Any, Any, Any, Any]: ...
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
        for BT in [2, 4, 8]
    ],
    key=["num_heads", "head_dim", "BLOCK_SIZE_D"],
    **autotune_cache_kwargs,
)
@triton.jit
def gate_output_correction_fwd_kernel(
    o_ptr,
    r_ptr,
    k_ptr,
    r_k_ptr,
    v_ptr,
    g_ptr,
    output_ptr,
    o_b_stride,
    o_t_stride,
    r_b_stride,
    r_t_stride,
    r_h_stride,
    v_b_stride,
    v_t_stride,
    v_h_stride,
    r_k_h_stride,
    T,
    T_OFFSET,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BT: tl.constexpr,
): ...
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 4]
        for BT in [2, 4, 8]
    ],
    key=["num_heads", "head_dim", "BLOCK_SIZE_D"],
    **autotune_cache_kwargs,
)
@triton.jit
def gate_output_correction_bwd_kernel(
    grad_output_ptr,
    o_ptr,
    r_ptr,
    k_ptr,
    r_k_ptr,
    v_ptr,
    g_ptr,
    grad_o_ptr,
    grad_r_ptr,
    grad_k_ptr,
    grad_r_k_intermediate_ptr,
    grad_v_ptr,
    grad_g_ptr,
    r_b_stride,
    r_t_stride,
    r_h_stride,
    o_b_stride,
    o_t_stride,
    r_k_h_stride,
    T,
    T_OFFSET,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BT: tl.constexpr,
): ...
def gate_output_correction_backward_triton(
    grad_output, o, r, k, r_k, v, g
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: ...

class GateOutputCorrection(torch.autograd.Function):
    @staticmethod
    @autocast_custom_fwd
    @input_guard
    def forward(ctx, o, r, k, r_k, v, g) -> Tensor: ...
    @staticmethod
    @autocast_custom_bwd
    @input_guard
    def backward(
        ctx, grad_output
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: ...

gate_output_correction = ...
