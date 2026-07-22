import torch
import triton
import triton.language as tl

@triton.heuristics({"IS_VARLEN": lambda args: args["offsets"] is not None})
@triton.jit(do_not_specialize=["T"])
def parallel_path_fwd_kernel_prepare_k_cache(
    k,
    k_new,
    w1,
    w2,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def prepare_k_cache_fn(
    k,
    w1,
    w2,
    cu_seqlens,
    BS,
    use_cache=...,
    chunk_indices: torch.LongTensor | None = ...,
) -> Tensor | None: ...
