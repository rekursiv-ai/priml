from dataclasses import dataclass

from torchao.dtypes.affine_quantized_tensor import register_layout
from torchao.dtypes.utils import AQTTensorImpl, Layout

import torch

aten = ...

@dataclass(frozen=True)
class MarlinSparseLayout(Layout):
    def pre_process(self, input: torch.Tensor) -> torch.Tensor: ...

@register_layout(MarlinSparseLayout)
class MarlinSparseAQTTensorImpl(AQTTensorImpl):
    @staticmethod
    def __new__(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        meta: torch.Tensor,
        _layout: Layout,
        original_shape: torch.Size,
        group_size: int,
        num_bits: int,
    ): ...
    def __init__(
        self,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        meta: torch.Tensor,
        _layout: Layout,
        original_shape: torch.Size,
        group_size: int,
        num_bits: int,
    ) -> None: ...
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
        ...
    def __tensor_flatten__(self):  # -> tuple[list[str], list[Layout | Size | int]]:
        ...
    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):  # -> Self:
        ...
    def get_plain(self):  # -> tuple[Tensor, Tensor, Tensor | Any]:
        ...
    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        _layout: Layout,
    ):  # -> Self:
        ...
    def get_layout(self) -> Layout: ...
