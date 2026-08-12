from typing import Any

from torchao.prototype.mx_formats.config import (
    MXFP8Dim1CastKernelChoice,
    MXGemmKernelChoice,
    MXLinearConfig,
    ScaleCalculationMode,
)

import torch

"""
Defines the prototype UX for converting a model to use mx weights
"""

@torch._dynamo.allow_in_graph
class mx_mm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp: torch.Tensor,
        in_elem_dtype: Any,
        w_elem_dtype: Any,
        grad_elem_dtype: Any,
        block_size: int,
        gemm_kernel_choice: MXGemmKernelChoice,
        mxfp8_cast_kernel_choice: MXFP8Dim1CastKernelChoice,
        scale_calculation_mode: ScaleCalculationMode,
    ):  # -> Tensor:
        ...
    @staticmethod
    def backward(
        ctx, grad_output_hp: torch.Tensor
    ):  # -> tuple[Tensor, Tensor, None, None, None, None, None, None, None]:
        ...

class MXLinear(torch.nn.Linear):
    @classmethod
    @torch.no_grad()
    def from_float(cls, mod, config: MXLinearConfig | None = ...):  # -> Linear:
        ...
    def forward(self, x):  # -> None:
        ...
    def extra_repr(self):  # -> str:
        ...
