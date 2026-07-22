from fla.ops.cp.context import FLACPContext
from fla.utils import USE_CUDA_GRAPH, autotune_cache_kwargs

import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_BG": lambda args: args["bg"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "HV", "K", "V", "BT", "USE_EXP2"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def pre_process_fwd_kernel_merged(
    k,
    v,
    w,
    g,
    gk,
    bg,
    u,
    hm,
    cu_seqlens,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_BG: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MULTI_SEQS: tl.constexpr,
): ...
@triton.heuristics({"HAS_H0": lambda args: args["h0"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=["HV", "K", "V", "BT", "USE_EXP2"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["pre_or_post_num_ranks", "rank", "NUM_SEQ_ENTRIES"])
def merge_fwd_bwd_kernel(
    h,
    ag_hm,
    pre_or_post_num_ranks,
    rank,
    seq_offsets,
    init_offsets,
    h0_seq_ids,
    h0,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    FORWARD: tl.constexpr,
    INTRACARD_MODE: tl.constexpr,
    NUM_SEQ_ENTRIES,
    HAS_H0: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr = ...,
) -> None: ...
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def pre_process_bwd_kernel_merged(
    q,
    k,
    w,
    g,
    gk,
    do,
    dhm,
    dv,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_BG: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def chunk_gated_delta_rule_fwd_h_pre_process(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = ...,
    gk: torch.Tensor | None = ...,
    bg: torch.Tensor | None = ...,
    v: torch.Tensor | None = ...,
    chunk_size: int = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    use_exp2: bool = ...,
    initial_state: torch.Tensor | None = ...,
    context: FLACPContext = ...,
    transpose_state_layout: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def chunk_gated_delta_rule_bwd_dhu_pre_process(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = ...,
    gk: torch.Tensor | None = ...,
    bg: torch.Tensor | None = ...,
    scale: float | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    use_exp2: bool = ...,
    dht: torch.Tensor | None = ...,
    initial_state: torch.Tensor | None = ...,
    context: FLACPContext | None = ...,
    transpose_state_layout: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def compress_h0(h0: torch.Tensor, context: FLACPContext) -> Tensor: ...
def expand_h0(h0: torch.Tensor, context: FLACPContext) -> Tensor: ...
