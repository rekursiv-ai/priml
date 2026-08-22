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
def parallel_path_fwd_kernel(
    q,
    k,
    v,
    o,
    o_new,
    g_cumsum,
    w1,
    w2,
    scale,
    L,
    L_new,
    M,
    cu_seqlens,
    indices,
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
    USE_GATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def parallel_path_fwd_fn(
    q,
    k,
    v,
    o,
    g_cumsum,
    w1,
    w2,
    scale,
    L,
    M,
    cu_seqlens,
    BT,
    BS,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Tensor, Tensor]: ...
