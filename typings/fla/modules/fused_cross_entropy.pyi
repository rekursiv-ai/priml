from torch import Tensor
from typing import Any

from fla.utils import input_guard
from torch import nn

import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "HAS_SMOOTHING": lambda args: args["label_smoothing"] > 0,
        "HAS_SOFTCAPPING": lambda args: args["logit_softcapping"] is not None,
    }
)
@triton.jit
def cross_entropy_fwd_kernel(
    loss_ptr,
    lse_ptr,
    z_loss_ptr,
    logits_ptr,
    labels_ptr,
    label_smoothing,
    logit_scale,
    lse_square_scale,
    logit_softcapping,
    ignore_index,
    total_classes,
    class_start_idx,
    n_cols,
    n_rows,
    logits_row_stride,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
    SPLIT: tl.constexpr,
): ...
@triton.heuristics(
    {
        "HAS_SMOOTHING": lambda args: args["label_smoothing"] > 0,
        "HAS_SOFTCAPPING": lambda args: args["logit_softcapping"] is not None,
    }
)
@triton.jit
def cross_entropy_bwd_kernel(
    dlogits_ptr,
    dloss_ptr,
    logits_ptr,
    lse_ptr,
    labels_ptr,
    label_smoothing,
    logit_scale,
    lse_square_scale,
    logit_softcapping,
    ignore_index,
    total_classes,
    class_start_idx,
    n_cols,
    logits_row_stride,
    dlogits_row_stride,
    dloss_row_stride,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
): ...
def fused_cross_entropy_forward(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = ...,
    logit_scale: float = ...,
    lse_square_scale: float = ...,
    logit_softcapping: float = ...,
    ignore_index: int = ...,
    process_group=...,
) -> tuple[Any, Tensor | Any, Tensor | Any, int, int]: ...

class CrossEntropyLossFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        logits,
        target,
        label_smoothing=...,
        logit_scale=...,
        lse_square_scale=...,
        logit_softcapping=...,
        ignore_index=...,
        inplace_backward=...,
        process_group=...,
    ) -> tuple[Any, Tensor | Any]: ...
    @staticmethod
    @input_guard
    def backward(
        ctx, grad_losses, grad_z_losses
    ) -> tuple[Any | Tensor, None, None, None, None, None, None, None, None, None]: ...

def cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = ...,
    logit_scale: float = ...,
    lse_square_scale: float = ...,
    logit_softcapping: float = ...,
    ignore_index=...,
    inplace_backward: bool = ...,
    process_group=...,
) -> tuple[torch.Tensor, torch.Tensor]: ...

class FusedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = ...,
        reduction: str = ...,
        label_smoothing: float = ...,
        logit_scale: float = ...,
        lse_square_scale: float = ...,
        logit_softcapping: float = ...,
        inplace_backward: bool = ...,
        process_group: Any = ...,
        return_z_loss: bool = ...,
    ) -> None: ...
    def forward(self, input, target) -> Tensor | tuple[Any | Tensor, Any | Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Tensor | tuple[Any | Tensor, Any | Tensor]: ...
