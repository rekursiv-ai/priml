from fla.utils import input_guard
from torch import Tensor

import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_GV": lambda args: args["gv"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_GATE_IN_KERNEL": lambda args: args["A_log"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    gk,
    gv,
    beta,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
): ...
def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = ...,
    gk: torch.Tensor | None = ...,
    gv: torch.Tensor | None = ...,
    beta: torch.Tensor | None = ...,
    A_log: torch.Tensor | None = ...,
    dt_bias: torch.Tensor | None = ...,
    scale: float = ...,
    initial_state: torch.Tensor = ...,
    output_final_state: bool = ...,
    use_qk_l2norm_in_kernel: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    use_exp2: bool = ...,
    transpose_state_layout: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...

class FusedRecurrentFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = ...,
        gk: torch.Tensor | None = ...,
        gv: torch.Tensor | None = ...,
        beta: torch.Tensor | None = ...,
        A_log: torch.Tensor | None = ...,
        dt_bias: torch.Tensor | None = ...,
        scale: float = ...,
        initial_state: torch.Tensor = ...,
        output_final_state: bool = ...,
        use_qk_l2norm_in_kernel: bool = ...,
        cu_seqlens: torch.LongTensor | None = ...,
        use_exp2: bool = ...,
        transpose_state_layout: bool = ...,
    ) -> tuple[Tensor, Tensor]: ...
    @staticmethod
    @input_guard
    def backward(ctx, do, dht): ...

def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = ...,
    gk: torch.Tensor | None = ...,
    gv: torch.Tensor | None = ...,
    beta: torch.Tensor | None = ...,
    scale: float = ...,
    initial_state: torch.Tensor = ...,
    output_final_state: bool = ...,
    use_qk_l2norm_in_kernel: bool = ...,
    use_gate_in_kernel: bool = ...,
    A_log: torch.Tensor | None = ...,
    dt_bias: torch.Tensor | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    use_exp2: bool = ...,
    transpose_state_layout: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...

fused_recurrent_gdn = ...
