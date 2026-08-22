from typing import Any

from fla.ops.backends import dispatch
from fla.ops.cp import FLACPContext
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from torch import Tensor

import torch

class ChunkKDAFunction(torch.autograd.Function):
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
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool = ...,
        use_qk_l2norm_in_kernel: bool = ...,
        use_gate_in_kernel: bool = ...,
        cu_seqlens: torch.LongTensor | None = ...,
        cu_seqlens_cpu: torch.LongTensor | None = ...,
        safe_gate: bool = ...,
        lower_bound: float | None = ...,
        disable_recompute: bool = ...,
        return_intermediate_states: bool = ...,
        cp_context: FLACPContext | None = ...,
        transpose_state_layout: bool = ...,
    ) -> tuple[Tensor, Tensor | None, Tensor | None] | tuple[Tensor, Tensor | None]: ...
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx, do: torch.Tensor, dht: torch.Tensor
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor | Any,
        Tensor | Any,
        Tensor | None,
        Tensor | None,
        None,
        Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]: ...

@dispatch("kda")
@torch.compiler.disable
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    use_qk_l2norm_in_kernel: bool = ...,
    use_gate_in_kernel: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
    safe_gate: bool = ...,
    lower_bound: float | None = ...,
    disable_recompute: bool = ...,
    return_intermediate_states: bool = ...,
    cp_context: FLACPContext = ...,
    transpose_state_layout: bool = ...,
    **kwargs,
) -> None: ...
