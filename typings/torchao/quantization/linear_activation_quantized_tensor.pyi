from collections.abc import Callable
from typing import Any

from torchao.utils import TorchAOBaseTensor

import torch

__all__ = ["LinearActivationQuantizedTensor", "to_linear_activation_quantized"]
aten = ...

class LinearActivationQuantizedTensor(TorchAOBaseTensor):
    quant_kwargs: dict[str, Any]
    def __new__(
        cls,
        original_weight_tensor: torch.Tensor,
        input_quant_func: Callable,
        quant_kwargs: dict[str, Any],
    ):  # -> Self:
        ...
    def __init__(
        self,
        original_weight_tensor: torch.Tensor,
        input_quant_func: Callable[[torch.Tensor], torch.Tensor],
        quant_kwargs: dict[str, Any],
    ) -> None: ...
    def __repr__(self):  # -> str:
        ...
    def __tensor_flatten__(
        self,
    ):  # -> tuple[list[str], list[Callable[[Tensor], Tensor] | Dict[str, Any]]]:
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
        quant_kwargs: dict[str, Any] | None = ...,
    ):  # -> Self:
        ...
    def to(self, *args, **kwargs):  # -> Self:
        ...

implements = ...

@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):  # -> Tensor:
    ...
@implements([aten.mm.default, aten.addmm.default])
def _(func, types, args, kwargs): ...
@implements([aten.detach.default, aten.alias.default])
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.clone.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten._to_copy.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.copy_.default)
def _(func, types, args, kwargs):  # -> None:
    ...
@implements(aten.t.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.select.int)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.index.Tensor)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.view.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...

to_linear_activation_quantized = ...
