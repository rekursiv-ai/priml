from dataclasses import dataclass

from torchao.dtypes.affine_quantized_tensor import register_layout
from torchao.dtypes.utils import AQTTensorImpl, Layout

import torch

aten = ...

@dataclass(frozen=True)
class Int4CPULayout(Layout): ...

@register_layout(Int4CPULayout)
class Int4CPUAQTTensorImpl(AQTTensorImpl):
    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scale_and_zero: torch.Tensor,
        transposed: bool,
        _layout: Layout,
    ):  # -> Self:
        ...
    def __init__(
        self,
        packed_weight: torch.Tensor,
        scale_and_zero: torch.Tensor,
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
        zero_point: torch.Tensor | None,
        _layout: Layout,
    ):  # -> Self:
        ...
    def to(self, *args, **kwargs):  # -> Self:
        ...
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
        ...

    __torch_function__ = ...
    @property
    def block_size(self):  # -> tuple[Literal[1], int]:
        ...
    def get_plain(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def get_layout(self) -> Layout: ...
