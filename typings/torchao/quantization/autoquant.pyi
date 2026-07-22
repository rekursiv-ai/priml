from torchao.dtypes import AffineQuantizedTensor
from torchao.dtypes.utils import Layout
from torchao.quantization.linear_activation_quantized_tensor import (
    LinearActivationQuantizedTensor,
)
from torchao.utils import TorchAOBaseTensor

import torch

__all__ = [
    "ALL_AUTOQUANT_CLASS_LIST",
    "DEFAULT_AUTOQUANT_CLASS_LIST",
    "DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST",
    "DEFAULT_INT4_AUTOQUANT_CLASS_LIST",
    "DEFAULT_SPARSE_AUTOQUANT_CLASS_LIST",
    "GEMLITE_INT4_AUTOQUANT_CLASS_LIST",
    "OTHER_AUTOQUANT_CLASS_LIST",
    "AutoQuantizableLinearWeight",
    "autoquant",
]
aten = ...
_AUTOQUANT_CACHE = ...

class AutoQuantizableLinearWeight(torch.Tensor):
    @staticmethod
    def __new__(
        cls, weight, qtensor_class_list, *args, mode=..., min_sqnr=..., **kwargs
    ): ...
    def __init__(
        self, weight, qtensor_class_list, *args, mode=..., min_sqnr=..., **kwargs
    ) -> None: ...
    def __repr__(self):  # -> str:
        ...
    @staticmethod
    def log_shape(act_mat, w_autoquant, bias):  # -> None:
        ...
    def tune_autoquant(self, q_cls, shapes_and_dtype, best_time):  # -> None:
        ...
    @torch.no_grad()
    def to_quantized(self, error_on_unseen, **kwargs): ...
    def __tensor_flatten__(
        self,
    ):  # -> tuple[list[str], list[Any | Callable[..., Any] | dtype | Size | None]]:
        ...
    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size=..., outer_stride=...
    ):  # -> Self:
        ...
    @classmethod
    def from_float(cls, weight, qtensor_class_list, **kwargs):  # -> Self:
        ...
    @classmethod
    def __torch_function__(cls, func, types, args=..., kwargs=...):  # -> None:
        ...
    @classmethod
    def __torch_dispatch__(
        cls, func, types, args, kwargs
    ):  # -> tuple[Any, ...] | Any | None:
        ...

@torch.no_grad()
def do_autoquant_bench(op, *args, **kwargs):  # -> float | list[float]:
    ...

class AQMixin: ...

class AQInt8DynamicallyQuantizedLinearWeight(AQMixin, LinearActivationQuantizedTensor):
    aq_layout: Layout = ...
    @classmethod
    def from_float(cls, weight):  # -> Self:
        ...

class AQInt8DynamicallyQuantizedSemiSparseLinearWeight(
    AQInt8DynamicallyQuantizedLinearWeight
):
    aq_layout: Layout = ...

class AQInt8WeightOnlyQuantizedLinearWeight(AffineQuantizedTensor, AQMixin):
    @classmethod
    def from_float(cls, weight):  # -> Self:
        ...

class AQInt8WeightOnlyQuantizedLinearWeight2(
    AQInt8WeightOnlyQuantizedLinearWeight, AQMixin
): ...
class AQInt8WeightOnlyQuantizedLinearWeight3(
    AQInt8WeightOnlyQuantizedLinearWeight, AQMixin
): ...

class AQInt4G32WeightOnlyQuantizedLinearWeight(
    LinearActivationQuantizedTensor, AQMixin
):
    group_size: int = ...
    aq_layout: Layout = ...
    @classmethod
    def from_float(cls, weight):  # -> Self:
        ...

class AQInt4G64WeightOnlyQuantizedLinearWeight(
    AQInt4G32WeightOnlyQuantizedLinearWeight
):
    group_size: int = ...

class AQInt4G128WeightOnlyQuantizedLinearWeight(
    AQInt4G32WeightOnlyQuantizedLinearWeight
):
    group_size: int = ...

class AQInt4G256WeightOnlyQuantizedLinearWeight(
    AQInt4G32WeightOnlyQuantizedLinearWeight
):
    group_size: int = ...

class AQInt4G128WeightOnlyQuantizedMarlinSparseLinearWeight(
    AQInt4G32WeightOnlyQuantizedLinearWeight
):
    group_size: int = ...
    aq_layout: Layout = ...

class AQGemliteInt4G32WeightOnlyQuantizedLinearWeight(
    LinearActivationQuantizedTensor, AQMixin
):
    group_size: int = ...
    @classmethod
    def from_float(cls, weight):  # -> Self:
        ...

class AQGemliteInt4G64WeightOnlyQuantizedLinearWeight(
    AQGemliteInt4G32WeightOnlyQuantizedLinearWeight
):
    group_size: int = ...

class AQGemliteInt4G128WeightOnlyQuantizedLinearWeight(
    AQGemliteInt4G32WeightOnlyQuantizedLinearWeight
):
    group_size: int = ...

class AQGemliteInt4G256WeightOnlyQuantizedLinearWeight(
    AQGemliteInt4G32WeightOnlyQuantizedLinearWeight
):
    group_size: int = ...

class AQDefaultLinearWeight(torch.Tensor, AQMixin):
    def __init__(self) -> None: ...
    @classmethod
    def from_float(cls, weight): ...

class Float32Tensor(TorchAOBaseTensor):
    @staticmethod
    def __new__(cls, weight, skip_weight_conversion=...): ...
    def __init__(self, weight, skip_weight_conversion=...) -> None: ...
    @classmethod
    def from_float(cls, weight):  # -> Self:
        ...

@Float32Tensor.implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs): ...
@Float32Tensor.implements(aten.detach.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@Float32Tensor.implements(aten.clone.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@Float32Tensor.implements(aten._to_copy.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...

class BFloat16Tensor(Float32Tensor):
    def __init__(self, weight, skip_weight_conversion=...) -> None: ...
    @classmethod
    def from_float(cls, weight, skip_weight_conversion=...):  # -> Self:
        ...

class Float16Tensor(Float32Tensor):
    def __init__(self, weight, skip_weight_conversion=...) -> None: ...
    @classmethod
    def from_float(cls, weight, skip_weight_conversion=...):  # -> Self:
        ...

class AQFloat32LinearWeight(Float32Tensor, AQMixin):
    @classmethod
    def from_float(cls, weight):  # -> Self:
        ...

class AQBFloat16LinearWeight(BFloat16Tensor, AQMixin):
    @classmethod
    def from_float(cls, weight):  # -> Self:
        ...

class AQFloat16LinearWeight(Float16Tensor, AQMixin):
    @classmethod
    def from_float(cls, weight):  # -> Self:
        ...

class AQFloat8WeightOnlyQuantizedLinearWeight(AffineQuantizedTensor, AQMixin):
    target_dtype: torch.dtype = ...
    @classmethod
    def from_float(cls, weight):  # -> Self:
        ...

class AQFloat8PerRowScalingDynamicallyQuantizedLinearWeight(
    AQMixin, LinearActivationQuantizedTensor
):
    activation_granularity = ...
    @classmethod
    def from_float(cls, weight):  # -> Self:
        ...

class AQFloat8PerTensorScalingDynamicallyQuantizedLinearWeight(
    AQMixin, LinearActivationQuantizedTensor
):
    activation_granularity = ...
    @classmethod
    def from_float(cls, weight):  # -> Self:
        ...

DEFAULT_AUTOQUANT_CLASS_LIST = ...
DEFAULT_INT4_AUTOQUANT_CLASS_LIST = ...
GEMLITE_INT4_AUTOQUANT_CLASS_LIST = ...
DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST = ...
OTHER_AUTOQUANT_CLASS_LIST = ...
DEFAULT_SPARSE_AUTOQUANT_CLASS_LIST = ...
ALL_AUTOQUANT_CLASS_LIST = ...
ALL_AUTOQUANT_CLASS_LIST = ...

@torch.no_grad()
def autoquant(
    model,
    example_input=...,
    qtensor_class_list=...,
    filter_fn=...,
    mode=...,
    manual=...,
    set_inductor_config=...,
    supress_autoquant_errors=...,
    min_sqnr=...,
    **aq_kwargs,
): ...
