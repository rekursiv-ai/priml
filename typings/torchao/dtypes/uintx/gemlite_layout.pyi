from dataclasses import dataclass

from torchao.dtypes.affine_quantized_tensor import register_layout
from torchao.dtypes.uintx.tensor_core_tiled_layout import TensorCoreTiledAQTTensorImpl
from torchao.dtypes.utils import Layout

import torch

aten = ...

def get_gemlite_quant_kwargs(bit_width, group_size, dtype):  # -> dict[Any, Any]:
    ...
def get_gemlite_aqt_kwargs(
    weight, group_size=..., bit_width=..., packing_bitwidth=..., mode=..., use_hqq=...
):  # -> dict[Any, Any]:
    ...

@dataclass(frozen=True)
class GemlitePackedLayout(Layout):
    group_size: int | None = ...
    bit_width: int = ...
    packing_bitwidth: int | None = ...
    mode: str | None = ...

@register_layout(GemlitePackedLayout)
class GemliteAQTTensorImpl(TensorCoreTiledAQTTensorImpl):
    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        gemlite_kwargs: dict,
        _layout: Layout,
    ):  # -> Self:
        ...
    def __init__(
        self,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        gemlite_kwargs: dict,
        _layout: Layout,
    ) -> None: ...
    def __tensor_flatten__(self):  # -> tuple[list[str], list[Layout | Dict[Any, Any]]]:
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
    def get_plain(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    @classmethod
    def __torch_dispatch__(
        cls, func, types, args, kwargs
    ):  # -> tuple[Any, ...] | Any | None:
        ...
    def get_layout(self) -> Layout: ...
    @property
    def block_size(self):  # -> tuple[Literal[1], Any]:
        ...
