from typing import Any

from torchao.float8.config import Float8LinearConfig
from torchao.float8.float8_training_tensor import LinearMMConfig

import torch

"""
A simple module swap UX for a float8 version of `torch.nn.Linear`.
"""

@torch._dynamo.allow_in_graph
class matmul_with_hp_or_float8_args(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp_t: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        config: Float8LinearConfig,
    ):  # -> Tensor:
        ...
    @staticmethod
    def backward(ctx, grad_output):  # -> tuple[Tensor, Tensor, None, None]:
        ...

class Float8Linear(torch.nn.Linear):
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def extra_repr(self):  # -> str:
        ...
    @classmethod
    def from_float(cls, mod, config: Float8LinearConfig | None = ...):  # -> Self:
        ...
