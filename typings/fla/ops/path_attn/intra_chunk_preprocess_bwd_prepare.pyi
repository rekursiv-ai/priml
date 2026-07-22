import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "USE_GATE": lambda args: args["g_cumsum"] is not None,
        "IS_VARLEN": lambda args: args["offsets"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_transform_qk_bwd_kernel_prepare(
    q,
    k,
    v,
    w,
    beta,
    g_cumsum,
    L,
    D,
    h,
    q_new,
    k_new,
    AT,
    dA_local,
    dv,
    do,
    dg_cumsum,
    scale,
    indices,
    offsets,
    chunk_offsets,
    T,
    G: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATE: tl.constexpr,
    RETURN_H: tl.constexpr,
) -> None: ...
def intra_chunk_preprocess_bwd_prepare_fn(
    q,
    k,
    v,
    w,
    beta,
    g_cumsum,
    A,
    L,
    D,
    do,
    scale,
    return_h=...,
    cu_seqlens=...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]: ...
