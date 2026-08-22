from typing import Any

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from torch import Tensor

import torch

def chunk_simple_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = ...,
    g_gamma: torch.Tensor | None = ...,
    scale: float | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def chunk_simple_gla_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_gamma: torch.Tensor,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class ChunkSimpleGLAFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        g_gamma,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
    ) -> tuple[Tensor, Tensor]: ...
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx, do, dht
    ) -> tuple[
        Tensor, Tensor, Tensor, Tensor | Any | None, None, None, Any, None, None, None
    ]: ...

@torch.compiler.disable
def chunk_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = ...,
    g_gamma: torch.Tensor | None = ...,
    scale: float | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]: ...
