from typing import Any

from torch import Tensor

import torch

class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, logits, l2_penalty_factor=...): ...
    @staticmethod
    def backward(ctx, grad_output) -> tuple[Any, Tensor, None]: ...

l2_warp = ...
