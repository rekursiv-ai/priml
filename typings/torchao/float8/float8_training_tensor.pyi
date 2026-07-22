from typing import NamedTuple

import enum

import torch

aten = ...

class ScaledMMConfig(NamedTuple):
    emulate: bool = ...
    use_fast_accum: bool = ...
    fp8_output: bool = ...
    pad_inner_dim: bool = ...

class LinearMMConfig(NamedTuple):
    output: ScaledMMConfig = ...
    grad_input: ScaledMMConfig = ...
    grad_weight: ScaledMMConfig = ...

class GemmInputRole(enum.Enum):
    INPUT = ...
    WEIGHT = ...
    GRAD_OUTPUT = ...

def choose_scaled_mm_config(
    a_role: GemmInputRole,
    a_linear_mm_config: LinearMMConfig,
    b_role: GemmInputRole,
    b_linear_mm_config: LinearMMConfig,
):  # -> ScaledMMConfig:
    ...

@torch._dynamo.allow_in_graph
class _ToFloat8ConstrFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        float8_dtype: torch.dtype,
        linear_mm_config: LinearMMConfig | None = ...,
        gemm_input_role: GemmInputRole | None = ...,
        axiswise_dim: int | None = ...,
    ):  # -> DTensor | Float8TrainingTensor:
        ...
    @staticmethod
    def backward(ctx, g):  # -> tuple[Any, None, None, None, None, None]:
        ...

@torch._dynamo.allow_in_graph
class _FromFloat8ConstrFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor): ...
    @staticmethod
    def backward(ctx, g):  # -> tuple[Any, None, None]:
        ...

def hp_tensor_and_scale_to_float8(
    hp_tensor: torch.Tensor,
    s: torch.Tensor,
    float8_dtype: torch.dtype,
    linear_mm_config: LinearMMConfig | None = ...,
    gemm_input_role: GemmInputRole | None = ...,
    axiswise_dim: int | None = ...,
):  # -> Any | None:
    ...

class Float8TrainingTensor(torch.Tensor):
    _data: torch.Tensor
    _scale: torch.Tensor
    _orig_dtype: torch.dtype
    _linear_mm_config: LinearMMConfig
    _gemm_input_role: GemmInputRole
    _axiswise_dim: int | None
    __slots__ = ...
    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        orig_dtype: torch.dtype,
        linear_mm_config: LinearMMConfig | None,
        gemm_input_role: GemmInputRole | None = ...,
        axiswise_dim: int | None = ...,
    ):  # -> Self:
        ...
    def __repr__(self):  # -> str:
        ...
    def __tensor_flatten__(
        self,
    ):  # -> tuple[list[str], dict[str, dtype | LinearMMConfig | GemmInputRole | int | None]]:
        ...
    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict, metadata, outer_size, outer_stride
    ):  # -> Float8TrainingTensor:
        ...
    def to_original_precision(self):  # -> Any | None:
        ...
    @classmethod
    def __torch_dispatch__(
        cls, func, types, args, kwargs=...
    ):  # -> _NotImplementedType | Any:
        ...

    __torch_function__ = ...
