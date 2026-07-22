from dataclasses import dataclass

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.utils import AQTTensorImpl, Layout
from torchao.quantization.quant_primitives import ZeroPointDomain

import torch

logger = ...
aten = ...

class MarlinQQQTensor(AffineQuantizedTensor):
    def dequantize(self, output_dtype: torch.dtype | None = ...) -> torch.Tensor: ...
    @classmethod
    def from_hp_to_intx(
        cls,
        input_float: torch.Tensor,
        block_size: tuple[int, ...],
        quant_min: int | None = ...,
        quant_max: int | None = ...,
        zero_point_domain: ZeroPointDomain = ...,
        _layout: Layout | None = ...,
    ):  # -> Self:
        ...

@dataclass(frozen=True)
class MarlinQQQLayout(Layout): ...

@register_layout(MarlinQQQLayout)
class MarlinQQQAQTTensorImpl(AQTTensorImpl):
    @staticmethod
    def __new__(
        cls,
        int_data: torch.Tensor,
        s_group: torch.Tensor,
        s_channel: torch.Tensor,
        _layout: Layout,
        original_shape: torch.Size,
        group_size: int,
        num_bits: int,
    ): ...
    def __init__(
        self,
        int_data: torch.Tensor,
        s_group: torch.Tensor,
        s_channel: torch.Tensor,
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
    def get_plain(self):  # -> tuple[Tensor, Tensor, Tensor]:
        ...
    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        s_group: torch.Tensor,
        s_channel: torch.Tensor,
        _layout: Layout,
    ):  # -> Self:
        ...
    def get_layout(self) -> Layout: ...

to_marlinqqq_quantized_intx = ...
