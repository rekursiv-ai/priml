from torchao.dtypes.utils import AQTTensorImpl, Layout
from torchao.quantization.quant_primitives import MappingType, ZeroPointDomain
from torchao.utils import TorchAOBaseTensor

import torch

logger = ...
aten = ...
__all__ = [
    "AffineQuantizedTensor",
    "register_layout",
    "to_affine_quantized_floatx",
    "to_affine_quantized_floatx_static",
    "to_affine_quantized_fpx",
    "to_affine_quantized_intx",
    "to_affine_quantized_intx_static",
]

class AffineQuantizedTensor(TorchAOBaseTensor):
    @staticmethod
    def __new__(
        cls,
        tensor_impl: AQTTensorImpl,
        block_size: tuple[int, ...],
        shape: torch.Size,
        quant_min: float | None = ...,
        quant_max: float | None = ...,
        zero_point_domain: ZeroPointDomain = ...,
        dtype=...,
        strides=...,
    ): ...
    def __init__(
        self,
        tensor_impl: AQTTensorImpl,
        block_size: tuple[int, ...],
        shape: torch.Size,
        quant_min: float | None = ...,
        quant_max: float | None = ...,
        zero_point_domain: ZeroPointDomain = ...,
        dtype=...,
        strides=...,
    ) -> None: ...
    def __repr__(self):  # -> str:
        ...
    def dequantize(self, output_dtype: torch.dtype | None = ...) -> torch.Tensor: ...
    def __tensor_flatten__(
        self,
    ):  # -> tuple[list[str], list[Tuple[int, ...] | Size | int | float | ZeroPointDomain | dtype | None]]:
        ...
    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):  # -> Self:
        ...
    @classmethod
    def from_hp_to_intx(
        cls,
        input_float: torch.Tensor,
        mapping_type: MappingType,
        block_size: tuple[int, ...],
        target_dtype: torch.dtype,
        quant_min: int | None = ...,
        quant_max: int | None = ...,
        eps: float | None = ...,
        scale_dtype: torch.dtype | None = ...,
        zero_point_dtype: torch.dtype | None = ...,
        preserve_zero: bool = ...,
        zero_point_domain: ZeroPointDomain = ...,
        _layout: Layout = ...,
        use_hqq: bool = ...,
        *,
        custom_scale: torch.Tensor | None = ...,
        custom_zero_point: torch.Tensor | None = ...,
    ):  # -> Self:
        ...
    @classmethod
    def from_hp_to_intx_static(
        cls,
        input_float: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        block_size: tuple[int, ...],
        target_dtype: torch.dtype,
        quant_min: int | None = ...,
        quant_max: int | None = ...,
        zero_point_domain: ZeroPointDomain = ...,
        _layout: Layout = ...,
    ):  # -> Self:
        ...
    @classmethod
    def from_hp_to_floatx(
        cls,
        input_float: torch.Tensor,
        block_size: tuple[int, ...],
        target_dtype: torch.dtype,
        _layout: Layout,
        scale_dtype: torch.dtype | None = ...,
    ):  # -> Self:
        ...
    @classmethod
    def from_hp_to_floatx_static(
        cls,
        input_float: torch.Tensor,
        scale: torch.Tensor,
        block_size: tuple[int, ...],
        target_dtype: torch.dtype,
        _layout: Layout,
        scale_dtype: torch.dtype = ...,
    ):  # -> Self:
        ...
    @classmethod
    def from_hp_to_fpx(cls, input_float: torch.Tensor, _layout: Layout):  # -> Self:
        ...
    def to(self, *args, **kwargs):  # -> Self:
        ...

register_layout = ...
get_tensor_impl_constructor = ...
to_affine_quantized_intx = ...
to_affine_quantized_intx_static = ...
to_affine_quantized_floatx = ...
to_affine_quantized_floatx_static = ...
to_affine_quantized_fpx = ...
