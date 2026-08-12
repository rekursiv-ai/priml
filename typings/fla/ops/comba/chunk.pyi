from torch import Tensor
from typing import Any
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

import torch

def chunk_comba_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Any, Any | Tensor, Tensor, Tensor | Any, Tensor | None]: ...
def chunk_comba_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g0: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | Any, Tensor]: ...

class ChunkCombaFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
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
        cu_seqlens_cpu: torch.LongTensor | None = ...,
    ) -> tuple[Tensor, Tensor | None]: ...
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
        Tensor | Any,
        Tensor,
        None,
        Tensor,
        None,
        None,
        None,
        None,
    ]: ...

@torch.compiler.disable
def chunk_comba(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = ...,
    scale: float = ...,
    initial_state: torch.Tensor = ...,
    output_final_state: bool = ...,
    use_qk_l2norm_in_kernel: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
) -> tuple[Any, Any]: ...
