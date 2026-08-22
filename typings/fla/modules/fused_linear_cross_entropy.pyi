from typing import Any

from fla.utils import input_guard
from torch import Tensor, nn
from torch.distributed.tensor.parallel import ParallelStyle

import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = ...
STATIC_WARPS = ...

@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "HAS_SOFTCAPPING": lambda args: args["softcapping"] is not None,
    }
)
@triton.jit
def logsumexp_fwd_kernel(
    x,
    z,
    scale,
    softcapping,
    D: tl.constexpr,
    B: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
): ...
def logsumexp_fwd(
    x,
    scale: float | None = ...,
    softcapping: float | None = ...,
    dtype: torch.dtype | None = ...,
): ...
@triton.jit
def cross_entropy_kernel(
    logits,
    lse,
    target,
    loss,
    total,
    ignore_index,
    label_smoothing: tl.constexpr,
    logit_scale: tl.constexpr,
    logit_softcapping: tl.constexpr,
    reduction: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
): ...
@triton.jit
def elementwise_mul_kernel(x, g, N: tl.constexpr, B: tl.constexpr) -> None: ...
def fused_linear_cross_entropy_forward(
    x: torch.Tensor,
    target: torch.LongTensor,
    weight: torch.Tensor,
    bias: torch.Tensor = ...,
    ignore_index: int = ...,
    label_smoothing: float = ...,
    logit_scale: float = ...,
    logit_softcapping: float = ...,
    num_chunks: int = ...,
    reduction: str = ...,
    use_l2warp: bool = ...,
    l2_penalty_factor: float = ...,
) -> tuple[Tensor, Tensor, Tensor | Any | None, Tensor | None]: ...
def fused_linear_cross_entropy_backward(
    do: torch.Tensor, dx: torch.Tensor, dw: torch.Tensor, db: torch.Tensor
) -> tuple[Tensor, Tensor, Tensor]: ...

class FusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        target: torch.LongTensor,
        weight: torch.Tensor,
        bias: torch.Tensor = ...,
        ignore_index: int = ...,
        label_smoothing: float = ...,
        logit_scale: float = ...,
        logit_softcapping: float = ...,
        num_chunks: int = ...,
        reduction: str = ...,
        use_l2warp: bool = ...,
        l2_penalty_factor: float = ...,
    ) -> Tensor: ...
    @staticmethod
    @input_guard
    def backward(
        ctx, do
    ) -> tuple[
        Tensor, None, Tensor, Tensor, None, None, None, None, None, None, None, None
    ]: ...

def fused_linear_cross_entropy_loss(
    x: torch.Tensor,
    target: torch.LongTensor,
    weight: torch.Tensor,
    bias: torch.Tensor = ...,
    ignore_index: int = ...,
    label_smoothing: float = ...,
    logit_scale: float = ...,
    logit_softcapping: float = ...,
    num_chunks: int = ...,
    reduction: str = ...,
    use_l2warp: bool = ...,
    l2_penalty_factor: float = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...

class FusedLinearCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = ...,
        label_smoothing: float = ...,
        logit_scale: float = ...,
        logit_softcapping: float = ...,
        num_chunks: int = ...,
        reduction: str = ...,
        use_l2warp: bool = ...,
        l2_penalty_factor: float = ...,
    ) -> None: ...
    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        target: torch.LongTensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = ...,
    ) -> tuple[Tensor, Tensor]: ...

class LinearLossParallel(ParallelStyle):
    def __init__(
        self, *, sequence_dim: int = ..., use_local_output: bool = ...
    ) -> None: ...
