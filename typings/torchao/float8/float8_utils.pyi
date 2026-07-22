from collections.abc import Iterable

from torchao.float8.config import ScalingGranularity

import torch

EPS = ...
IS_ROCM = ...
FP8_TYPES = ...

@torch.no_grad()
def amax_to_scale(
    amax: torch.Tensor,
    float8_dtype: torch.dtype,
    round_scales_to_power_of_2: bool = ...,
):  # -> Tensor:
    ...
@torch.no_grad()
def tensor_to_amax(
    x: torch.Tensor,
    reduce_amax: bool = ...,
    device_mesh=...,
    scaling_granularity: ScalingGranularity = ...,
    axiswise_dim: int | None = ...,
) -> torch.Tensor: ...
@torch.no_grad()
def tensor_to_scale(
    hp_tensor: torch.Tensor,
    float8_dtype: torch.dtype,
    reduce_amax: bool = ...,
    device_mesh=...,
    scaling_granularity: ScalingGranularity = ...,
    axiswise_dim: int | None = ...,
    round_scales_to_power_of_2: bool = ...,
) -> torch.Tensor: ...
def to_fp8_saturated(x: torch.Tensor, float8_dtype: torch.dtype):  # -> Tensor:
    ...
def compute_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...
def fp8_tensor_statistics(
    tensor: torch.Tensor, float8_dtype: torch.dtype
) -> tuple[int, ...]: ...
def is_row_major(stride): ...
def pad_tensor_for_matmul(
    tensor: torch.Tensor, dims: int | Iterable[int]
) -> torch.Tensor: ...
