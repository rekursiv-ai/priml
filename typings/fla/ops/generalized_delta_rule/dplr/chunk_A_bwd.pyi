from torch import Tensor
from fla.utils import USE_CUDA_GRAPH, autotune_cache_kwargs

import torch
import triton
import triton.language as tl

NUM_WARPS_AUTOTUNE = ...

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
    ],
    key=["BK", "BT", "K"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_dplr_bwd_kernel_intra(
    q,
    k,
    a,
    b,
    gi,
    ge,
    dAqk,
    dAqb,
    dAak,
    dAab,
    dq,
    dk,
    da,
    db,
    dqg,
    dkg,
    dag,
    dbg,
    dgk,
    dgk_offset,
    cu_seqlens,
    chunk_indices,
    scale: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    GATHER_SUPPORTED: tl.constexpr,
) -> None: ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BK", "BT"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_dplr_bwd_kernel_intra_tensorcore(
    q,
    k,
    a,
    b,
    gi,
    ge,
    dAqk,
    dAqb,
    dAak,
    dAab,
    dq,
    dk,
    da,
    db,
    dqg,
    dkg,
    dag,
    dbg,
    dgk,
    dgk_offset,
    cu_seqlens,
    chunk_indices,
    scale: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    GATHER_SUPPORTED: tl.constexpr,
) -> None: ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
        for BK in [32, 64]
    ],
    key=["BK", "BT", "K"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_dplr_bwd_dgk_kernel(
    dgk,
    dgk_offset,
    dgk_last,
    dgk_output,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def chunk_dplr_bwd_dqk_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    dAqk: torch.Tensor,
    dAqb: torch.Tensor,
    dAak: torch.Tensor,
    dAab: torch.Tensor,
    dqg: torch.Tensor,
    dkg: torch.Tensor,
    dag: torch.Tensor,
    dbg: torch.Tensor,
    dgk_last: torch.Tensor,
    scale: float = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    safe_gate: bool = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: ...
