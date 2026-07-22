from dataclasses import dataclass

from torchao.dtypes.affine_quantized_tensor import register_layout
from torchao.dtypes.uintx.plain_layout import PlainAQTTensorImpl
from torchao.dtypes.utils import Layout
from torchao.utils import TorchAOBaseTensor

import torch

aten = ...
_DTYPE_TO_BIT_WIDTH = ...
_BIT_WIDTH_TO_DTYPE = ...
_DTYPE_TO_BIT_WIDTH = ...
_BIT_WIDTH_TO_DTYPE = ...

class UintxTensor(TorchAOBaseTensor):
    bits_to_shard = ...
    def __new__(
        cls,
        shards: list[torch.Tensor],
        packed_shape: list[int],
        bit_width: int,
        pack_dim: int = ...,
    ):  # -> Self:
        ...
    def __init__(
        self,
        shards: list[torch.Tensor],
        packed_shape: list[int],
        bit_width: int,
        pack_dim: int = ...,
    ) -> None: ...
    def get_shards(self):  # -> list[Any]:
        ...
    def __repr__(self):  # -> str:
        ...
    def __tensor_flatten__(self):  # -> tuple[list[str], list[List[int] | int]]:
        ...
    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):  # -> Self:
        ...
    def get_plain(self):  # -> Tensor:
        ...
    def apply_transformation(self, fn):  # -> Self:
        ...
    def apply_fn_to_shards(self, fn):  # -> Self:
        ...
    @classmethod
    def from_uint8(
        cls, int_data: torch.Tensor, dtype: torch.dtype, pack_dim: int = ...
    ):  # -> Self:
        ...
    def to(self, *args, **kwargs):  # -> Tensor | Self:
        ...

implements = ...

@implements(aten.detach.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.view.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten._to_copy.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.sub.Tensor)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.mul.Tensor)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...

to_uintx = ...

@dataclass(frozen=True)
class UintxLayout(Layout):
    dtype: torch.dtype
    pack_dim: int = ...
    def post_process(
        self,
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

@register_layout(UintxLayout)
class UintxAQTTensorImpl(PlainAQTTensorImpl):
    def get_plain(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        _layout: Layout,
    ):  # -> Self:
        ...
