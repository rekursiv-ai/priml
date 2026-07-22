from dataclasses import dataclass

from torchao.float8.inference import Float8MMConfig, FP8Granularity
from torchao.quantization.quantize_.common import KernelPreference, QuantizeTensorKwargs
from torchao.utils import TorchAOBaseTensor

import torch

__all__ = ["Float8Tensor", "QuantizeTensorToFloat8Kwargs"]
aten = ...

@dataclass
class QuantizeTensorToFloat8Kwargs(QuantizeTensorKwargs):
    float8_dtype: torch.dtype = ...
    granularity: FP8Granularity = ...
    mm_config: Float8MMConfig | None = ...
    hp_value_lb: float | None = ...
    hp_value_ub: float | None = ...
    kernel_preference: KernelPreference = ...

class Float8Tensor(TorchAOBaseTensor):
    tensor_data_names = ...
    tensor_attribute_names = ...
    optional_tensor_attribute_names = ...
    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_size: list[int] | None = ...,
        mm_config: Float8MMConfig | None = ...,
        act_quant_kwargs: QuantizeTensorToFloat8Kwargs | None = ...,
        kernel_preference: KernelPreference = ...,
        dtype: torch.dtype | None = ...,
    ):  # -> Self:
        ...
    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_size: list[int] | None = ...,
        mm_config: Float8MMConfig | None = ...,
        act_quant_kwargs: QuantizeTensorToFloat8Kwargs | None = ...,
        kernel_preference: KernelPreference = ...,
        dtype: torch.dtype | None = ...,
    ) -> None: ...
    def __repr__(self):  # -> str:
        ...
    def dequantize(self, output_dtype: torch.dtype | None = ...) -> torch.Tensor: ...
    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        float8_dtype: torch.dtype = ...,
        granularity: FP8Granularity = ...,
        mm_config: Float8MMConfig | None = ...,
        hp_value_lb: float | None = ...,
        hp_value_ub: float | None = ...,
        kernel_preference: KernelPreference = ...,
        act_quant_kwargs: QuantizeTensorToFloat8Kwargs | None = ...,
    ):  # -> Float8Tensor:
        ...

implements = ...

@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):  # -> Any | Tensor:
    ...
@implements(torch.bmm)
def _(func, types, args, kwargs):  # -> Any:
    ...
@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.cat.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.transpose.int)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.view.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.squeeze.dim)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.select.int)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.unsqueeze.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
