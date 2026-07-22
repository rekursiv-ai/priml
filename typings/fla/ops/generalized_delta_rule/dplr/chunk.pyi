from fla.ops.cp import FLACPContext
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

import torch

def chunk_dplr_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    safe_gate: bool = ...,
    chunk_indices: torch.LongTensor | None = ...,
    disable_recompute: bool = ...,
    cp_context: FLACPContext | None = ...,
) -> (
    tuple[
        Tensor,
        Any,
        Tensor,
        tuple[
            Any,
            Any,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
        ],
    ]
    | tuple[Tensor, Any, Tensor, None]
): ...

class ChunkDPLRDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        gk: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = ...,
        cu_seqlens_cpu: torch.LongTensor | None = ...,
        safe_gate: bool = ...,
        chunk_size: int | None = ...,
        disable_recompute: bool = ...,
        cp_context: FLACPContext | None = ...,
    ) -> tuple[Tensor, Any]: ...
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx, do: torch.Tensor, dht: torch.Tensor
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        None,
        Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]: ...

@torch.compiler.disable
def chunk_dplr_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
    safe_gate: bool = ...,
    chunk_size: int | None = ...,
    disable_recompute: bool = ...,
    cp_context: FLACPContext | None = ...,
    **kwargs,
) -> tuple[Any, Any]: ...
