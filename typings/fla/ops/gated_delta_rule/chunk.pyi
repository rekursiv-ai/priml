from typing import Any

from fla.ops.cp import FLACPContext
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

import torch

def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = ...,
    cp_context: FLACPContext | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
    use_exp2: bool = ...,
    transpose_state_layout: bool = ...,
    use_gate_in_kernel: bool = ...,
    A_log: torch.Tensor | None = ...,
    dt_bias: torch.Tensor | None = ...,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
]: ...
def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = ...,
    cp_context: FLACPContext | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
    use_exp2: bool = ...,
    transpose_state_layout: bool = ...,
    use_gate_in_kernel: bool = ...,
    g_input: torch.Tensor | None = ...,
    A_log: torch.Tensor | None = ...,
    dt_bias: torch.Tensor | None = ...,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]: ...

class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = ...,
        cu_seqlens_cpu: torch.LongTensor | None = ...,
        use_qk_l2norm_in_kernel: bool = ...,
        cp_context: FLACPContext | None = ...,
        transpose_state_layout: bool = ...,
        use_gate_in_kernel: bool = ...,
        A_log: torch.Tensor | None = ...,
        dt_bias: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx, do: torch.Tensor, dht: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]: ...

@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    use_qk_l2norm_in_kernel: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
    cp_context: FLACPContext | None = ...,
    transpose_state_layout: bool = ...,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]: ...

chunk_gdn = ...
