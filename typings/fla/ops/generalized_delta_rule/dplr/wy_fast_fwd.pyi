from fla.utils import USE_CUDA_GRAPH, autotune_cache_kwargs

import torch
import triton
import triton.language as tl

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16]],
    key=["BT"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def prepare_wy_repr_fwd_kernel_chunk32(
    A_ab,
    A_ab_inv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
) -> None: ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BC"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def prepare_wy_repr_fwd_kernel_chunk64(
    A_ab,
    A_ab_inv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    GATHER_SUPPORTED: tl.constexpr = ...,
): ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def wu_fwd_kernel(
    w,
    u,
    ag,
    v,
    A_ab_inv,
    A_ak,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def wu_fwd(
    ag: torch.Tensor,
    v: torch.Tensor,
    A_ak: torch.Tensor,
    A_ab_inv: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def prepare_wy_repr_fwd(
    ag: torch.Tensor,
    v: torch.Tensor,
    A_ak: torch.Tensor,
    A_ab: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

fwd_prepare_wy_repr = ...
fwd_wu = ...
