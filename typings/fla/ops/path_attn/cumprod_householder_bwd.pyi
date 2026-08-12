from torch import Tensor
import torch
import triton
import triton.language as tl

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_cumprod_householder_bwd_kernel(
    hc_suffix,
    dhc_whole,
    k,
    dk,
    w1,
    w2,
    dw1,
    dw2,
    dk_new,
    cu_seqlens,
    split_indices,
    chunk_offsets,
    split_offsets,
    BT: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    G: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def chunk_cumprod_householder_bwd_fn(
    w1: torch.Tensor,
    w2: torch.Tensor,
    hc_suffix: torch.Tensor,
    dhc_whole: torch.Tensor,
    k: torch.Tensor,
    dk: torch.Tensor,
    S: int,
    BT: int,
    cu_seqlens: torch.Tensor = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...
