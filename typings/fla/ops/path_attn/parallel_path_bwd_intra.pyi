from torch import Tensor
from typing import Any
import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["offsets"] is not None,
        "USE_GATE": lambda args: args["g_cumsum"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_path_bwd_intra_chunk_kernel(
    q,
    k,
    v,
    g_cumsum,
    w1,
    w2,
    L,
    D,
    dq,
    dq_new,
    dk,
    dv,
    dw1,
    dw2,
    do,
    dg_cumsum,
    offsets,
    indices,
    T,
    scale,
    G: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    S: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATE: tl.constexpr,
): ...
def parallel_path_bwd_intra_chunk_fn(
    q,
    k,
    v,
    g_cumsum,
    w1,
    w2,
    dq,
    dk,
    dv,
    dg_cumsum,
    dw1,
    dw2,
    do,
    scale,
    L,
    D,
    cu_seqlens,
    S,
    BT,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Tensor, Any, Any, Any, Any, Any]: ...
