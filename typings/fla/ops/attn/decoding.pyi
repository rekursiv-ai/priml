import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "USE_G": lambda args: args["g_cumsum"] is not None,
        "USE_SINK_BIAS": lambda args: args["sink_bias"] is not None,
    }
)
@triton.jit
def naive_attn_decoding_kernel(
    q,
    k,
    v,
    o,
    g_cumsum,
    sink_bias,
    scale,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_SINK_BIAS: tl.constexpr,
): ...
def attn_decoding_one_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = ...,
    scale: float | None = ...,
    cu_seqlens: torch.LongTensor = ...,
    do_gate_scale: bool = ...,
    *,
    sink_bias: torch.Tensor | None = ...,
) -> Tensor: ...
