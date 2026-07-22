from collections.abc import Callable
from typing import Any

from torchao.utils import TorchAOBaseTensor

import torch

__all__ = [
    ...,
    ...,
]
aten = ...

class WeightTensorWithLinearActivationQuantizationMetadata(TorchAOBaseTensor):
    original_weight_tensor: torch.Tensor
    input_quant_func_static: Callable
    scale: torch.Tensor
    zero_point: torch.Tensor | None
    quant_kwargs: dict[str, Any]
    def __new__(
        cls,
        original_weight_tensor: torch.Tensor,
        input_quant_func_static: Callable,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        quant_kwargs: dict[str, Any],
    ):  # -> Self:
        ...
    def __init__(
        self,
        original_weight_tensor: torch.Tensor,
        input_quant_func_static: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]],
            torch.Tensor,
        ],
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        quant_kwargs: dict[str, Any],
    ) -> None: ...
    def __repr__(self):  # -> str:
        ...
    def __tensor_flatten__(
        self,
    ):  # -> tuple[list[str], list[Callable[..., Any] | Dict[str, Any]]]:
        ...
    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):  # -> Self:
        ...
    @classmethod
    def from_float(
        cls,
        input_float: torch.Tensor,
        input_quant_func: Callable,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None = ...,
        quant_kwargs: dict[str, Any] | None = ...,
    ):  # -> Self:
        ...
    def to(self, *args, **kwargs):  # -> Self:
        ...

implements = ...

@implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):  # -> Tensor:
    ...
@implements(aten.detach.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.clone.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten._to_copy.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.t.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...

to_weight_tensor_with_linear_activation_quantization_metadata = ...
