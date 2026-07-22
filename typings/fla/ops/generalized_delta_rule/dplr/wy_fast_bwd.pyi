from fla.utils import USE_CUDA_GRAPH, autotune_cache_kwargs

import torch
import triton
import triton.language as tl

triton_config = ...

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config(triton_config, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "BK", "BV"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def prepare_wy_repr_bwd_kernel(
    A_ab_inv,
    A_ak,
    ag,
    v,
    dw,
    du,
    dv,
    dv0,
    dag,
    dAak,
    dAab,
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
def chunk_dplr_bwd_wy(
    A_ab_inv: torch.Tensor,
    A_ak: torch.Tensor,
    v: torch.Tensor,
    ag: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    dv0: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
