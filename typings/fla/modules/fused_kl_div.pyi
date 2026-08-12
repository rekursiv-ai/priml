from typing import Any
from torch import Tensor
from fla.utils import input_guard
from torch import nn

import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = ...
STATIC_WARPS = ...

@triton.jit
def kl_div_kernel(
    logits,
    target_logits,
    loss,
    s_logits,
    s_loss,
    reduction: tl.constexpr,
    N: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
): ...
@triton.jit
def elementwise_mul_kernel(x, g, N: tl.constexpr, B: tl.constexpr) -> None: ...
def fused_kl_div_forward(
    x: torch.Tensor,
    target_x: torch.Tensor,
    weight: torch.Tensor,
    target_weight: torch.Tensor,
    reduction: str = ...,
) -> tuple[Tensor, Tensor, Tensor | None]: ...
def fused_kl_div_backward(
    do: torch.Tensor, dx: torch.Tensor, dw: torch.Tensor
) -> tuple[Tensor, Tensor]: ...

class FusedKLDivLossFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        target_x: torch.Tensor,
        weight: torch.Tensor,
        target_weight: torch.Tensor,
        reduction: str,
    ) -> Tensor: ...
    @staticmethod
    @input_guard
    def backward(ctx, do) -> tuple[Tensor, None, Tensor, None, None]: ...

def fused_kl_div_loss(
    x: torch.Tensor,
    target_x: torch.Tensor,
    weight: torch.Tensor,
    target_weight: torch.Tensor,
    reduction: str = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...

class FusedKLDivLoss(nn.Module):
    def __init__(self, reduction: str = ...) -> None: ...
    def forward(
        self,
        x: torch.Tensor,
        target_x: torch.Tensor,
        weight: torch.Tensor,
        target_weight: torch.Tensor,
    ) -> tuple[Tensor, Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]: ...
