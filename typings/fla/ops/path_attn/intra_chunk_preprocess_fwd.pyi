from torch import Tensor
import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "USE_G": lambda args: args["g_cumsum"] is not None,
        "IS_VARLEN": lambda args: args["offsets"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def intra_chunk_preprocess_fwd_kernel(
    q,
    k,
    v,
    w,
    beta,
    g_cumsum,
    o,
    A,
    L,
    M,
    w2,
    q_new,
    k_new,
    scale,
    indices,
    offsets,
    T,
    H: tl.constexpr,
    G: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
): ...
def intra_chunk_preprocess_fwd_fn(
    q,
    k,
    v,
    w,
    beta,
    g_cumsum,
    A,
    scale,
    BT,
    cu_seqlens,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: ...
