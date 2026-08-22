from fla.utils import input_guard
from torch import Tensor

import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_comba_fwd_kernel(
    q,
    k,
    p,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_EXP2: tl.constexpr,
): ...
def fused_recurrent_comba_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    use_exp2: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...

class FusedRecurrentCombaFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        p: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = ...,
        cu_seqlens: torch.LongTensor | None = ...,
    ) -> tuple[Tensor, Tensor]: ...
    @staticmethod
    @input_guard
    def backward(ctx, do, dht): ...

def fused_recurrent_comba(
    q: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = ...,
    scale: float = ...,
    initial_state: torch.Tensor = ...,
    output_final_state: bool = ...,
    use_qk_l2norm_in_kernel: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
