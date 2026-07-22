from dataclasses import dataclass

from torchao.dtypes.affine_quantized_tensor import register_layout
from torchao.dtypes.utils import AQTTensorImpl, Layout

import torch

aten = ...

@dataclass(frozen=True)
class CutlassSemiSparseLayout(Layout):
    def pre_process(self, dense: torch.Tensor) -> torch.Tensor: ...

@register_layout(CutlassSemiSparseLayout)
class CutlassSemiSparseTensorImpl(AQTTensorImpl):
    @staticmethod
    def __new__(
        cls,
        sparse: torch.Tensor,
        meta: torch.Tensor,
        scale: torch.Tensor,
        _layout: Layout,
    ): ...
    def __init__(
        self,
        sparse: torch.Tensor,
        meta: torch.Tensor,
        scale: torch.Tensor,
        _layout: Layout,
    ) -> None: ...
    @classmethod
    def __torch_dispatch__(
        cls, func, types, args, kwargs
    ):  # -> tuple[Any, ...] | Any | None:
        ...
    def __tensor_flatten__(self):  # -> tuple[list[str], list[Layout]]:
        ...
    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):  # -> Self:
        ...
    def get_plain(self):  # -> tuple[Tensor, Tensor | Any, None]:
        ...
    @classmethod
    def from_plain(
        cls,
        dense: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        _layout: Layout,
    ):  # -> Self:
        ...
    def get_layout(self) -> Layout: ...
