from fla.ops.backends import dispatch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous

import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "USE_G": lambda args: args["g_cumsum"] is not None,
        "USE_SINK_BIAS": lambda args: args["sink_bias"] is not None,
        "USE_WINDOW": lambda args: args["W"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit
def parallel_attn_fwd_kernel(
    q,
    k,
    v,
    o,
    g_cumsum,
    sink_bias,
    lse,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    W: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_SINK_BIAS: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.jit
def parallel_attn_bwd_kernel_preprocess(
    o, do, delta, B: tl.constexpr, V: tl.constexpr
): ...
@triton.heuristics(
    {
        "USE_G": lambda args: args["g_cumsum"] is not None,
        "USE_WINDOW": lambda args: args["W"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_attn_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    dg_cumsum,
    g_cumsum,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    W: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics(
    {
        "USE_G": lambda args: args["g_cumsum"] is not None,
        "USE_WINDOW": lambda args: args["W"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_attn_bwd_kernel_dkv(
    q,
    k,
    v,
    g_cumsum,
    lse,
    delta,
    do,
    dk,
    dv,
    dg_cumsum,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    W: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@dispatch("attn")
def parallel_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    sink_bias: torch.Tensor | None,
    scale: float,
    window_size: int | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Tensor, Tensor]: ...
def parallel_attn_bwd_preprocess(o: torch.Tensor, do: torch.Tensor) -> Tensor: ...
@dispatch("attn")
def parallel_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    g_cumsum: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    sink_bias: torch.Tensor | None = ...,
    scale: float = ...,
    window_size: int | None = ...,
    chunk_size: int = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]: ...

@torch.compile
class ParallelAttentionFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(
        ctx, q, k, v, g, sink_bias, scale, window_size, cu_seqlens, chunk_indices=...
    ) -> Tensor: ...
    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(
        ctx, do
    ) -> tuple[
        Tensor, Tensor, Tensor, Tensor | Any, Tensor | None, None, None, None, None
    ]: ...

def parallel_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = ...,
    scale: float | None = ...,
    window_size: int | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
    *,
    sink_bias: torch.Tensor | None = ...,
    **kwargs,
) -> torch.Tensor: ...
