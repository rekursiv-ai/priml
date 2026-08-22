from typing import Any

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from torch import Tensor

import torch
import triton
import triton.language as tl

@triton.jit
def fused_recurrent_gsa_inference_kernel(
    q,
    k,
    v,
    s,
    g,
    o,
    hk0,
    hv0,
    hkt,
    hvt,
    scale,
    K: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
): ...
def fused_recurrent_gsa_inference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = ...,
    output_final_state: bool = ...,
    scale: float = ...,
) -> torch.Tensor: ...
def fused_recurrent_gsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = ...,
    output_final_state: bool = ...,
    scale: float = ...,
    reverse: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, tuple[torch.Tensor]]: ...
def fused_recurrent_gsa_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    qv: torch.Tensor,
    hk0: torch.Tensor | None = ...,
    hv0: torch.Tensor | None = ...,
    ok: torch.Tensor | None = ...,
    do: torch.Tensor | None = ...,
    dhkt: torch.Tensor | None = ...,
    dhvt: torch.Tensor | None = ...,
    scale: float = ...,
    reverse: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor]: ...

class FusedRecurrentGSAFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        s: torch.Tensor,
        g: torch.Tensor,
        scale: float | None = ...,
        hk0: torch.Tensor | None = ...,
        hv0: torch.Tensor | None = ...,
        output_final_state: bool = ...,
        reverse: bool = ...,
        cu_seqlens: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor]]: ...
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx, do, dhkt=..., dhvt=...
    ) -> tuple[Tensor, Any, Any, Any, Any, None, Any, Any, None, None, None]: ...

def fused_recurrent_gsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor | None = ...,
    scale: int | None = ...,
    initial_state: tuple[torch.Tensor] | None = ...,
    output_final_state: bool | None = ...,
    reverse: bool | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
