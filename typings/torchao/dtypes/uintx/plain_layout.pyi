from torchao.dtypes.affine_quantized_tensor import register_layout
from torchao.dtypes.utils import AQTTensorImpl, Layout, PlainLayout

import torch

aten = ...

@register_layout(PlainLayout)
class PlainAQTTensorImpl(AQTTensorImpl):
    def __new__(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        _layout: Layout,
    ):  # -> Self:
        ...
    def __init__(
        self,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        _layout: Layout,
    ) -> None: ...
    def __tensor_flatten__(self):  # -> tuple[list[str], list[Layout]]:
        ...
    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):  # -> Self:
        ...
    def to(self, *args, **kwargs):  # -> Self:
        ...
    @classmethod
    def __torch_dispatch__(
        cls, func, types, args, kwargs
    ):  # -> tuple[Any, ...] | Any | PlainAQTTensorImpl | None:
        ...

    __torch_function__ = ...
    def get_plain(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...
    def get_layout(self) -> Layout: ...
    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        _layout: Layout,
    ):  # -> Self:
        ...
