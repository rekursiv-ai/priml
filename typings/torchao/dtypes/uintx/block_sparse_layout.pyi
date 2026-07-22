from dataclasses import dataclass

from torchao.dtypes.affine_quantized_tensor import register_layout
from torchao.dtypes.uintx.plain_layout import PlainAQTTensorImpl
from torchao.dtypes.utils import Layout

import torch

logger = ...
aten = ...

@dataclass(frozen=True)
class BlockSparseLayout(Layout):
    blocksize: int = ...

@register_layout(BlockSparseLayout)
class BlockSparseAQTTensorImpl(PlainAQTTensorImpl):
    bsr_crow_indices: torch.Tensor | None
    bsr_col_indices: torch.Tensor | None
    bsr_values: torch.Tensor | None
    scale: torch.Tensor | None
    zero_point: torch.Tensor | None
    __slots__ = ...
    @staticmethod
    def __new__(
        cls,
        shape: torch.Size,
        bsr_crow_indices: torch.Tensor | None,
        bsr_col_indices: torch.Tensor | None,
        bsr_values: torch.Tensor | None,
        scale: torch.Tensor | None,
        zero_point: torch.Tensor | None,
        _layout: Layout,
        requires_grad: bool = ...,
    ): ...
    def __init__(
        self,
        shape: torch.Size,
        bsr_crow_indices: torch.Tensor | None,
        bsr_col_indices: torch.Tensor | None,
        bsr_values: torch.Tensor | None,
        scale: torch.Tensor | None,
        zero_point: torch.Tensor | None,
        _layout: Layout,
        requires_grad: bool = ...,
    ) -> None: ...
    def __tensor_flatten__(self):  # -> tuple[list[str], tuple[Size, Layout, bool]]:
        ...
    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors,
        tensor_meta: tuple[torch.Size, bool],
        outer_size,
        outer_stride,
    ) -> torch.Tensor: ...
    @classmethod
    def from_plain(cls, int_data, scale, zero_point, _layout):  # -> Self:
        ...
    def get_plain(self):  # -> tuple[Any, Tensor | None, Tensor | None]:
        ...
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
        ...
