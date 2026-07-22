from fla.utils import autotune_cache_kwargs, input_guard, tensor_cache

import torch
import triton
import triton.language as tl

NUM_WARPS_AUTOTUNE = ...

def token_shift_ref(
    x: torch.Tensor, cu_seqlens: torch.Tensor | None = ...
) -> torch.Tensor: ...
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_INITIAL_STATE": lambda args: args["cache"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [1, 2, 3]
    ],
    key=["BD"],
    **autotune_cache_kwargs,
)
@triton.jit
def token_shift_fwd_kernel_short(
    x,
    y,
    cu_seqlens,
    cache,
    cache_out,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_DECODE: tl.constexpr,
) -> None: ...
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_INITIAL_STATE": lambda args: args["cache"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [1, 2, 3]
    ],
    key=["BD", "NB"],
    **autotune_cache_kwargs,
)
@triton.jit
def token_shift_fwd_kernel_long(
    x,
    y,
    cu_seqlens,
    chunk_indices,
    cache,
    cache_out,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    NB: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
) -> None: ...
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_INITIAL_STATE": lambda args: args["grad_cache_out"] is not None,
        "HAS_DCACHE": lambda args: args["grad_cache_in"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [1, 2, 3]
    ],
    key=["BD"],
    **autotune_cache_kwargs,
)
@triton.jit
def token_shift_bwd_kernel_short(
    dx,
    dy,
    cu_seqlens,
    grad_cache_in,
    grad_cache_out,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    HAS_DCACHE: tl.constexpr,
) -> None: ...
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_INITIAL_STATE": lambda args: args["grad_cache_out"] is not None,
        "HAS_DCACHE": lambda args: args["grad_cache_in"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [1, 2, 3]
    ],
    key=["BD", "NB"],
    **autotune_cache_kwargs,
)
@triton.jit
def token_shift_bwd_kernel_long(
    dx,
    dy,
    cu_seqlens,
    chunk_indices,
    grad_cache_in,
    grad_cache_out,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    NB: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    HAS_DCACHE: tl.constexpr,
) -> None: ...
@tensor_cache
def prepare_maxlens(cu_seqlens: torch.LongTensor) -> int: ...
def token_shift_fwd(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor | None = ...,
    cache: torch.Tensor | None = ...,
    output_cache: bool = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> torch.Tensor: ...
def token_shift_bwd(
    dy: torch.Tensor,
    N: int,
    T: int,
    dcache: torch.Tensor | None = ...,
    cu_seqlens: torch.Tensor | None = ...,
    use_short_kernel: bool = ...,
    has_init_cache: bool = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> torch.Tensor: ...

class TokenShift(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = ...,
        cache: torch.Tensor | None = ...,
        output_cache: bool = ...,
        chunk_indices: torch.LongTensor | None = ...,
    ) -> tuple[Any, Any]: ...
    @staticmethod
    @input_guard
    def backward(
        ctx, dy: torch.Tensor, dcache: torch.Tensor | None = ...
    ) -> tuple[Any, None, Any, None, None]: ...

def token_shift(
    x: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = ...,
    cache: torch.Tensor | None = ...,
    output_cache: bool = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Any, Any] | Any: ...
