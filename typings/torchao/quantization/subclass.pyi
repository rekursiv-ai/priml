import torch

__all__ = [
    "Int4WeightOnlyQuantizedLinearWeight",
    "Int8DynamicallyQuantizedLinearWeight",
    "Int8WeightOnlyQuantizedLinearWeight",
]
aten = ...

class QuantizedLinearWeightBase(torch.Tensor):
    @staticmethod
    def __new__(cls, int_data, transposed, shape, *args, **kwargs): ...
    def __init__(self, int_data, transposed, *args, **kwargs) -> None: ...
    def __repr__(self):  # -> str:
        ...
    def dequantize(self):  # -> None:
        ...
    def int_repr(self):  # -> None:
        ...
    def q_params(self):  # -> None:
        ...
    def half(self):  # -> Tensor:
        ...
    def __tensor_flatten__(self):  # -> None:
        ...
    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):  # -> None:
        ...
    @classmethod
    def from_float(cls, input_float):  # -> None:
        ...
    @classmethod
    def __torch_function__(cls, func, types, args=..., kwargs=...):  # -> None:
        ...
    @classmethod
    def __torch_dispatch__(
        cls, func, types, args, kwargs
    ):  # -> tuple[Any, ...] | Any | None:
        ...

class ConstructTensorSubclass(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(self, x):  # -> None:
        ...
    def right_inverse(self, tensor_subclass_instance):  # -> list[Any]:
        ...

@torch._dynamo.allow_in_graph
def from_qtensor_components_int8dyn(
    *args, **kwargs
):  # -> Int8DynamicallyQuantizedLinearWeight:
    ...

class ConstructTensorSubclassInt8Dyn(ConstructTensorSubclass):
    def forward(self, int_data, q_scales):  # -> Int8DynamicallyQuantizedLinearWeight:
        ...

class Int8DynamicallyQuantizedLinearWeight(QuantizedLinearWeightBase):
    subclass_constructor = ConstructTensorSubclassInt8Dyn
    @staticmethod
    def __new__(cls, int_data, q_scales, transposed, shape, dtype=..., **kwargs): ...
    def __init__(
        self, int_data, q_scales, transposed, shape, dtype=..., **kwargs
    ) -> None: ...
    def dequantize(self, dtype=...):  # -> Tensor:
        ...
    def int_repr(self):  # -> Any:
        ...
    def q_params(self):  # -> dict[str, Any]:
        ...
    def to(self, *args, **kwargs):  # -> Self:
        ...
    def __tensor_flatten__(self):  # -> tuple[list[str], list[Any | Size | dtype]]:
        ...
    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size=..., outer_stride=...
    ):  # -> Self:
        ...
    @classmethod
    def from_float(cls, input_float, qmin=..., qmax=..., dtype=...):  # -> Self:
        ...

@torch._dynamo.allow_in_graph
def from_qtensor_components_int8wo(
    *args, **kwargs
):  # -> Int8WeightOnlyQuantizedLinearWeight:
    ...

class ConstructTensorSubclassInt8wo(ConstructTensorSubclass):
    def forward(self, int_data, q_scales):  # -> Int8WeightOnlyQuantizedLinearWeight:
        ...

class Int8WeightOnlyQuantizedLinearWeight(Int8DynamicallyQuantizedLinearWeight):
    subclass_constructor = ConstructTensorSubclassInt8wo

@torch._dynamo.allow_in_graph
def from_qtensor_components_int4wo(
    *args, **kwargs
):  # -> Int4WeightOnlyQuantizedLinearWeight:
    ...

class ConstructTensorSubclassInt4wo(ConstructTensorSubclass):
    def forward(
        self, int_data, scales_and_zeros
    ):  # -> Int4WeightOnlyQuantizedLinearWeight:
        ...

class Int4WeightOnlyQuantizedLinearWeight(QuantizedLinearWeightBase):
    subclass_constructor = ConstructTensorSubclassInt4wo
    @staticmethod
    def __new__(
        cls,
        int_data,
        scales_and_zeros,
        transposed,
        shape,
        groupsize=...,
        inner_k_tiles=...,
        zero_point_domain=...,
        preserve_zero=...,
        dtype=...,
        **kwargs,
    ): ...
    def __init__(
        self,
        int_data,
        scales_and_zeros,
        transposed,
        shape,
        groupsize,
        inner_k_tiles,
        zero_point_domain,
        preserve_zero,
        dtype,
        **kwargs,
    ) -> None: ...
    def dequantize(self):  # -> Any:
        ...
    def int_repr(self):  # -> Any:
        ...
    def q_params(self):  # -> dict[str, Tensor]:
        ...
    def to(self, *args, **kwargs):  # -> Self:
        ...
    def __tensor_flatten__(
        self,
    ):  # -> tuple[list[str], tuple[Any, Size, Any, Any, Any, Any, dtype]]:
        ...
    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, attributes, outer_size=..., outer_stride=...
    ):  # -> Self:
        ...
    @classmethod
    def from_float(
        cls,
        input_float,
        groupsize=...,
        inner_k_tiles=...,
        zero_point_domain=...,
        preserve_zero=...,
        dtype=...,
    ):  # -> Self:
        ...
    @classmethod
    def to_qtensor_components(
        cls,
        input_float,
        groupsize=...,
        inner_k_tiles=...,
        zero_point_domain=...,
        preserve_zero=...,
    ):  # -> tuple[Any, Tensor, Literal[False], int, int]:
        ...
