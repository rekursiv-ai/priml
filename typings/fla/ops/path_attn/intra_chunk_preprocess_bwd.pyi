from torch import Tensor
import torch
import triton
import triton.language as tl

@triton.heuristics({"IS_VARLEN": lambda args: args["offsets"] is not None})
@triton.jit(do_not_specialize=["T"])
def intra_chunk_preprocess_bwd_kernel(
    q,
    k,
    w,
    w2,
    beta,
    AT,
    dA_local,
    dq,
    dq_new,
    dk,
    dk_new,
    dw,
    dbeta,
    dw1,
    dw2,
    T,
    offsets,
    indices,
    HQ: tl.constexpr,
    G: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def intra_chunk_preprocess_bwd_fn(
    q,
    k,
    w,
    w2,
    beta,
    dq,
    dk,
    dA_local,
    dw1,
    dw2,
    A,
    L,
    D,
    do,
    scale,
    cu_seqlens=...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...
