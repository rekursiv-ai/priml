from fla.utils import autotune_cache_kwargs, input_guard

import torch
import triton
import triton.language as tl

NUM_WARPS_AUTOTUNE = ...

def k_update_ref(
    k: torch.Tensor, a: torch.Tensor, ka: torch.Tensor
) -> torch.Tensor: ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in NUM_WARPS_AUTOTUNE
        for s in [1, 2, 3]
    ],
    key=["BD"],
    **autotune_cache_kwargs,
)
@triton.jit
def k_update_fwd_kernel_short(
    k, a, ka, out, cu_seqlens, T, D, BD: tl.constexpr, IS_VARLEN: tl.constexpr
) -> None: ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in NUM_WARPS_AUTOTUNE
        for s in [1, 2, 3]
    ],
    key=["BD", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit
def k_update_fwd_kernel_long(
    k,
    a,
    ka,
    out,
    cu_seqlens,
    chunk_indices,
    T,
    D,
    BD: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
) -> None: ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=w, num_stages=s)
        for w in NUM_WARPS_AUTOTUNE
        for s in [1, 2, 3]
        for BT in [2, 4, 8]
    ],
    key=["BD"],
    **autotune_cache_kwargs,
)
@triton.jit
def k_update_bwd_kernel_short(
    grad_out,
    k,
    a,
    ka,
    dk,
    da,
    dka,
    cu_seqlens,
    T,
    D,
    BT: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
) -> None: ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in NUM_WARPS_AUTOTUNE
        for s in [1, 2, 3]
    ],
    key=["BD", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit
def k_update_bwd_kernel_long(
    grad_out,
    k,
    a,
    ka,
    dk,
    da,
    dka,
    cu_seqlens,
    chunk_indices,
    T,
    D,
    BD: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
) -> None: ...
def k_update_fwd(
    k: torch.Tensor,
    a: torch.Tensor,
    ka: torch.Tensor,
    cu_seqlens: torch.Tensor | None = ...,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
) -> torch.Tensor: ...
def k_update_bwd(
    grad_out: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    ka: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    use_short: bool,
    N: int,
    T: int,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...

class KUpdateFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(ctx, k, a, ka, cu_seqlens=..., cu_seqlens_cpu=...): ...
    @staticmethod
    @input_guard
    def backward(ctx, grad_output) -> tuple[Tensor, Tensor, Tensor, None, None]: ...

def fused_k_rwkv7(
    k, a, ka, cu_seqlens=..., cu_seqlens_cpu=...
) -> Tensor | Any | None: ...
