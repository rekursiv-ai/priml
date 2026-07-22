import enum

from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.quantize_.workflows.intx.intx_choose_qparams_algorithm import (
    IntxChooseQParamsAlgorithm,
)
from torchao.utils import TorchAOBaseTensor

import torch

__all__ = ["IntxUnpackedToInt8Tensor"]
aten = ...
_FLOAT_TYPES: list[torch.dtype] = ...

class IntxUnpackedToInt8TensorActivationQuantization(enum.StrEnum):
    INT8_ASYM_PER_TOKEN = ...

class IntxUnpackedToInt8Tensor(TorchAOBaseTensor):
    tensor_data_names = ...
    tensor_attribute_names = ...
    def __new__(
        cls,
        qdata,
        scale,
        zero_point,
        target_dtype,
        block_size,
        dtype,
        activation_quantization,
    ):  # -> Self:
        ...
    def __init__(
        self,
        qdata,
        scale,
        zero_point,
        target_dtype,
        block_size,
        dtype,
        activation_quantization,
    ) -> None: ...
    def to(self, *args, **kwargs):  # -> IntxUnpackedToInt8Tensor:
        ...
    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        block_size: tuple[int],
        target_dtype: torch.dtype,
        *,
        mapping_type: MappingType = ...,
        activation_quantization: IntxUnpackedToInt8TensorActivationQuantization
        | None = ...,
        intx_choose_qparams_algorithm: IntxChooseQParamsAlgorithm | None = ...,
        custom_scale: torch.Tensor | None = ...,
        custom_zero_point: torch.Tensor | None = ...,
    ):  # -> IntxUnpackedToInt8Tensor:
        ...
    def dequantize(self):  # -> Tensor:
        ...

implements = ...

@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):  # -> Tensor:
    ...
@implements([torch.nn.functional.embedding, aten.embedding.default])
def _(func, types, args, kwargs):  # -> Tensor:
    ...
@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
