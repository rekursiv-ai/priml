from fla.utils import IS_TF32_SUPPORTED, autotune_cache_kwargs

import torch
import triton
import triton.language as tl

if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = ...
else:
    SOLVE_TRIL_DOT_PRECISION = ...

@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4]
    ],
    key=["H", "HV", "K", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kkt_solve_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
) -> None: ...
def chunk_gated_delta_rule_fwd_intra(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = ...,
    beta: torch.Tensor | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    chunk_indices: torch.LongTensor | None = ...,
    use_exp2: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
