import torch
import triton
import triton.language as tl

int8_mm_kernel_configs = ...
if torch._inductor.config.max_autotune_gemm_search_space == "EXHAUSTIVE":
    int8_mm_kernel_configs = ...
int8_mm_kernel_configs = ...

@triton.jit
def matmul_kernel_with_block_pointers(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
): ...
@triton.jit
def scaled_matmul_kernel_with_block_pointers(
    a_ptr,
    b_ptr,
    c_ptr,
    s1_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_s1m,
    stride_s1n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACC_TYPE: tl.constexpr = ...,
): ...
def int_matmul_kernel(a, b, c, config): ...
def int_scaled_matmul_kernel(a, b, scales1, c, config): ...

lib = ...

@torch.library.impl(lib, "int_matmul", "Meta")
def int_matmul_meta(a, b):  # -> Tensor:
    ...
@torch.library.impl(lib, "int_matmul", "CUDA")
def int_matmul_cuda(a, b):  # -> Tensor:
    ...
@torch.library.impl(lib, "int_scaled_matmul", "Meta")
def int_scaled_matmul_meta(a, b, scales1):  # -> Tensor:
    ...
@torch.library.impl(lib, "int_scaled_matmul", "CUDA")
def int_scaled_matmul_cuda(a, b, scales1):  # -> Tensor:
    ...
