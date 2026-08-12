from torch import Tensor
from typing import Any
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

import torch

class ParallelPATHAttentionFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx, q, k, v, w, beta, g, scale, cu_seqlens, use_cache=...
    ) -> tuple[Tensor, Tensor | None]: ...
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx, do, dk_new
    ) -> tuple[
        Tensor, Tensor, Any, Tensor, Tensor, Tensor | Any | None, None, None, None, None
    ]: ...

@torch.compiler.disable
def parallel_path_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor | None = ...,
    scale: float = ...,
    cu_seqlens: torch.Tensor | None = ...,
    use_cache: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...

parallel_path_attention = ...
