from torchao.utils import TorchAOBaseTensor

import torch

__all__ = [
    "WeightTensorWithLinearActivationScaleMetadata",
    ...,
]
aten = ...

class WeightTensorWithLinearActivationScaleMetadata(TorchAOBaseTensor):
    tensor_data_names = ...
    tensor_attribute_names = ...
    def __new__(
        cls, original_weight_tensor: torch.Tensor, scale: torch.Tensor
    ):  # -> Self:
        ...
    def __init__(
        self, original_weight_tensor: torch.Tensor, scale: torch.Tensor
    ) -> None: ...
    @classmethod
    def from_float(cls, input_float: torch.Tensor, scale: torch.Tensor):  # -> Self:
        ...

implements = ...

@implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):  # -> Tensor:
    ...
@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.t.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...

to_weight_tensor_with_linear_activation_scale_metadata = ...
