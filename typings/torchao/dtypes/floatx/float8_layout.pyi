from dataclasses import dataclass
from typing import Any

from torchao.dtypes.affine_quantized_tensor import register_layout
from torchao.dtypes.utils import AQTTensorImpl, Layout
from torchao.float8.inference import Float8MMConfig

import torch

aten = ...
FLOAT8_IMPL_OPS_TABLE: dict[Any, Any] = ...

def implements(aten_ops: list[Any]):  # -> Callable[..., Any]:
    ...

@dataclass(frozen=True)
class Float8Layout(Layout):
    mm_config: Float8MMConfig | None = ...

_fallback_warning_shown = ...

@register_layout(Float8Layout)
class Float8AQTTensorImpl(AQTTensorImpl):
    float8_data: torch.Tensor
    scale: torch.Tensor
    transposed: bool
    def __new__(
        cls,
        float8_data: torch.Tensor,
        scale: torch.Tensor,
        transposed: bool,
        _layout: Layout,
    ):  # -> Self:
        ...
    def __init__(
        self,
        float8_data: torch.Tensor,
        scale: torch.Tensor,
        transposed: bool,
        _layout: Layout,
    ) -> None: ...
    def to(self, *args, **kwargs):  # -> Self:
        ...
    def __tensor_flatten__(self):  # -> tuple[list[str], list[bool | Layout]]:
        ...
    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):  # -> Self:
        ...
    @classmethod
    def __torch_dispatch__(
        cls, func, types, args, kwargs
    ):  # -> _NotImplementedType | Any:
        ...

    __torch_function__ = ...
    def get_plain(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...
    def get_layout(self) -> Layout: ...
    @classmethod
    def from_plain(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        _layout: Layout,
    ):  # -> Self:
        ...
    def __repr__(self):  # -> str:
        ...

@implements([aten.detach.default, aten.alias.default, aten.clone.default])
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements([aten.t.default])
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements([aten.copy_.default])
def _(func, types, args, kwargs):  # -> None:
    ...
@implements([aten.select.int, aten.index.Tensor])
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements([aten.slice.Tensor])
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
