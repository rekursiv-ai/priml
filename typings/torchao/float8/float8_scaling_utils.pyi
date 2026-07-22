from torchao.float8.config import ScalingGranularity
from torchao.float8.float8_training_tensor import (
    Float8TrainingTensor,
    GemmInputRole,
    LinearMMConfig,
)

import torch

"""
Utilities for scaling high precision tensors to float8.
"""

def hp_tensor_to_float8_dynamic(
    hp_tensor: torch.Tensor,
    float8_dtype: torch.dtype,
    linear_mm_config: LinearMMConfig,
    reduce_amax: bool = ...,
    gemm_input_role: GemmInputRole = ...,
    device_mesh=...,
    scaling_granularity: ScalingGranularity = ...,
    axiswise_dim: int | None = ...,
    round_scales_to_power_of_2: bool = ...,
) -> Float8TrainingTensor: ...
def get_maybe_axiswise_dim(
    axiswise_dim: int, scaling_granularity: ScalingGranularity
) -> int | None: ...

@torch._dynamo.allow_in_graph
class NoopFwToFloat8BwDynamic(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, tensor, linear_mm_config: LinearMMConfig, target_dtype: torch.dtype
    ): ...
    @staticmethod
    def backward(
        ctx, gradY
    ):  # -> tuple[Any, None, None] | tuple[Any | None, None, None]:
        ...
