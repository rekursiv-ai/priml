import torch
import triton
import triton.language as tl

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def transform_q_fwd_kernel(
    q,
    q_new,
    w1,
    w2,
    cu_seqlens,
    indices,
    T,
    S: tl.constexpr,
    G: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def transform_q_fwd_fn(
    q, w1, w2, cu_seqlens, BT, BS, S, chunk_indices: torch.LongTensor | None = ...
) -> Tensor: ...
