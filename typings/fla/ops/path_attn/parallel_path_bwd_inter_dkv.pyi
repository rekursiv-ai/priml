from typing import Any

from torch import Tensor

import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "USE_GATE": lambda args: args["g_cumsum"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_path_bwd_dkv_kernel(
    q,
    k,
    v,
    g_cumsum,
    hc_whole,
    scale,
    L,
    D,
    dk,
    dv,
    do,
    dg_cumsum,
    cu_seqlens,
    indices,
    split_offsets,
    T,
    G: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    S: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
): ...
def parallel_path_bwd_dkv_fn(
    q,
    k,
    v,
    g_cumsum,
    do,
    dv,
    dg_cumsum,
    hc_whole,
    scale,
    L,
    D,
    cu_seqlens,
    S,
    BT,
    BS,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Tensor, Any, Any]: ...
