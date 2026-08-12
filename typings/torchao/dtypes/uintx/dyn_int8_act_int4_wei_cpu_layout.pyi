from dataclasses import dataclass

from torchao.dtypes.affine_quantized_tensor import register_layout
from torchao.dtypes.utils import Layout

import torch

from .int4_cpu_layout import Int4CPUAQTTensorImpl

aten = ...

@dataclass(frozen=True)
class Int8DynamicActInt4WeightCPULayout(Layout): ...

@register_layout(Int8DynamicActInt4WeightCPULayout)
class DA8W4CPUAQTTensorImpl(Int4CPUAQTTensorImpl):
    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        compensation: torch.Tensor,
        transposed: bool,
        _layout: Layout,
    ):  # -> Self:
        ...
    def __init__(
        self,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        compensation: torch.Tensor,
        transposed: bool,
        _layout: Layout,
    ) -> None: ...
    def __tensor_flatten__(self):  # -> tuple[list[str], list[bool | Layout]]:
        ...
    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):  # -> Self:
        ...
    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        _layout: Layout,
    ):  # -> Self:
        ...
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
        ...
    @property
    def block_size(self):  # -> tuple[Literal[1], int]:
        ...
    def get_plain(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
