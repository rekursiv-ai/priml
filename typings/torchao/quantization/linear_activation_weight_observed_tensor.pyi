from torchao.quantization.observer import AffineQuantizedObserverBase
from torchao.utils import TorchAOBaseTensor

import torch

__all__ = ["LinearActivationWeightObservedTensor"]
aten = ...
Tensor = torch.Tensor

class LinearActivationWeightObservedTensor(TorchAOBaseTensor):
    original_weight_tensor: torch.Tensor
    input_observer: AffineQuantizedObserverBase | None
    weight_observer: AffineQuantizedObserverBase | None
    def __new__(
        cls,
        original_weight_tensor: torch.Tensor,
        input_observer: AffineQuantizedObserverBase | None = ...,
        weight_observer: AffineQuantizedObserverBase | None = ...,
    ):  # -> Self:
        ...
    def __init__(
        self,
        original_weight_tensor: torch.Tensor,
        input_observer: AffineQuantizedObserverBase | None = ...,
        weight_observer: AffineQuantizedObserverBase | None = ...,
    ) -> None: ...
    def __repr__(self):  # -> str:
        ...
    def __tensor_flatten__(self):  # -> tuple[list[str], list[Any | None]]:
        ...
    @classmethod
    def __tensor_unflatten__(
        cls,
        tensor_data_dict: dict[str, Tensor],
        tensor_attributes,
        outer_size,
        outer_stride,
    ):  # -> Self:
        ...
    @classmethod
    def from_float(
        cls,
        original_weight_tensor: Tensor,
        input_observer: AffineQuantizedObserverBase | None = ...,
        weight_observer: AffineQuantizedObserverBase | None = ...,
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
