from typing import Any

from torch import Tensor

import torch
import triton
import triton.language as tl

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit
def chunk_cumprod_householder_fwd_kernel(
    k,
    k_new,
    w1,
    w2,
    hc_suffix,
    hc_whole,
    cu_seqlens,
    split_indices,
    chunk_offsets,
    split_offsets,
    BT: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    BK: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def chunk_cumprod_householder_fwd_fn(
    k: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    S: int,
    BT: int,
    cu_seqlens: torch.Tensor = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Tensor, Any, Any]: ...
