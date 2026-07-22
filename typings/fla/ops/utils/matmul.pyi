from fla.utils import autotune_cache_kwargs, input_guard

import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "HAS_ALPHA": lambda args: args["alpha"] is not None,
        "HAS_BETA": lambda args: args["beta"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config(
            {"BM": 128, "BK": 64, "BN": 256, "G": 4}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BM": 64, "BK": 32, "BN": 256, "G": 4}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BM": 128, "BK": 32, "BN": 128, "G": 4}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BM": 128, "BK": 32, "BN": 64, "G": 4}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BM": 64, "BK": 32, "BN": 128, "G": 4}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BM": 128, "BK": 32, "BN": 32, "G": 4}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BM": 64, "BK": 32, "BN": 32, "G": 4}, num_stages=5, num_warps=2
        ),
        triton.Config(
            {"BM": 32, "BK": 32, "BN": 64, "G": 4}, num_stages=5, num_warps=2
        ),
    ],
    key=["M", "N", "K"],
    **autotune_cache_kwargs,
)
@triton.jit
def matmul_kernel(
    a,
    b,
    c,
    input,
    alpha,
    beta,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    G: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_INPUT: tl.constexpr,
    HAS_ALPHA: tl.constexpr,
    HAS_BETA: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    X_DIM: tl.constexpr = ...,
): ...
@triton.jit
def leaky_relu(x): ...
@triton.jit
def sigmoid(x): ...
@triton.jit
def tanh(x): ...
@triton.jit
def relu(x): ...
@input_guard
def matmul(a, b, activation=...): ...
@input_guard
def addmm(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float | None = ...,
    beta: float | None = ...,
) -> torch.Tensor: ...
