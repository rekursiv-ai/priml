from enum import Enum

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.utils import AQTTensorImpl, Layout

import torch

logger = ...
handler = ...
formatter = ...

class Target(Enum):
    AUTO = ...
    UNIVERSAL = ...
    KLEIDIAI = ...
    ATEN = ...

_TARGET_AND_STR = ...

def target_to_str(target: Target) -> str: ...
def target_from_str(target: str) -> Target: ...

class PackedLinearInt8DynamicActivationIntxWeightLayout(Layout):
    bit_width: int | None
    group_size: int | None
    has_weight_zeros: bool | None
    has_bias: bool | None
    target: Target | None
    def __init__(self, target: str | Target = ...) -> None: ...
    def extra_repr(self):  # -> str:
        ...
    def has_params_set(self) -> bool: ...
    def set_params(
        self, bit_width: int, group_size: int, has_weight_zeros: bool, has_bias: bool
    ):  # -> None:
        ...

@register_layout(PackedLinearInt8DynamicActivationIntxWeightLayout)
class PackedLinearInt8DynamicActivationIntxWeightAQTTensorImpl(AQTTensorImpl):
    def __new__(cls, packed_weight: torch.Tensor, _layout: Layout):  # -> Self:
        ...
    def __init__(self, packed_weight: torch.Tensor, _layout: Layout) -> None: ...
    def __repr__(self):  # -> str:
        ...
    def get_layout(self) -> Layout: ...
    def get_plain(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...
    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        layout: Layout,
        bias: torch.Tensor | None = ...,
        *,
        validate_inputs: bool = ...,
    ):  # -> Self:
        ...
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
        ...
    def __tensor_flatten__(self):  # -> tuple[list[str], list[Layout]]:
        ...
    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):  # -> Self:
        ...

def make_packed_linear_int8_dynamic_activation_intx_weight_tensor(
    int_data: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    bias: torch.Tensor | None,
    data_dtype: torch.dtype,
    target: str | Target = ...,
    *,
    validate_inputs: bool = ...,
) -> AffineQuantizedTensor: ...
