from collections.abc import Callable

from torchao.quantization.quant_primitives import MappingType, ZeroPointDomain
from torchao.utils import TorchAOBaseTensor

import torch

aten = ...

class _ToAffineFakeQuantized(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        original_tensor: torch.Tensor,
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
    ) -> _AffineFakeQuantizedTensor: ...
    @staticmethod
    def backward(
        ctx, gy
    ):  # -> tuple[Any, None, None, None, None, None, None, None, None, None, None]:
        ...

class _AffineFakeQuantizedTensor(TorchAOBaseTensor):
    @staticmethod
    def __new__(
        cls,
        original_tensor: torch.Tensor,
        apply_fake_quant_fn: Callable,
        fake_quant_enabled: bool = ...,
        **kwargs,
    ): ...
    def __init__(
        self,
        original_tensor: torch.Tensor,
        apply_fake_quant_fn: Callable,
        fake_quant_enabled: bool = ...,
        **kwargs,
    ) -> None: ...
    def __tensor_flatten__(
        self,
    ):  # -> tuple[list[str], list[Callable[..., Any] | bool]]:
        ...
    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):  # -> Self:
        ...
    @classmethod
    def from_float(
        cls,
        original_input: torch.Tensor,
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
    ):  # -> None:
        ...
    def get_value(self) -> torch.Tensor: ...
    def to(self, *args, **kwargs):  # -> Self:
        ...

implements = ...

@implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):  # -> Tensor:
    ...
@implements(aten.mm.default)
def _(func, types, args, kwargs): ...
@implements(aten.addmm.default)
def _(func, types, args, kwargs): ...
@implements(aten.detach.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.clone.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.t.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements([aten.add.Tensor, aten.add_.Tensor, aten.mul_.Tensor, aten.copy_.default])
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.empty_like.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.split.Tensor)
def _(func, types, args, kwargs):  # -> list[Any | tuple[Any, ...]]:
    ...
@implements(aten.new_zeros.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...

_to_affine_fake_quantized = ...
