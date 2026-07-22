from dataclasses import dataclass

from torchao.dtypes.affine_quantized_tensor import register_layout
from torchao.dtypes.uintx.plain_layout import PlainAQTTensorImpl
from torchao.dtypes.utils import Layout

import torch

aten = ...

@dataclass(frozen=True)
class SemiSparseLayout(Layout):
    def pre_process(self, input: torch.Tensor) -> torch.Tensor: ...

@register_layout(SemiSparseLayout)
class SemiSparseAQTTensorImpl(PlainAQTTensorImpl):
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
        ...
    def get_plain(self):  # -> tuple[Tensor, Tensor, Tensor | None]:
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
