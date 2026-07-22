from typing import Any

import torch

from .quant_primitives import MappingType
from .unified import Quantizer

aten = ...
__all__ = [
    "Int4WeightOnlyQuantizer",
    "Int8DynActInt4WeightQuantizer",
    "WeightOnlyInt4Linear",
]

def linear_forward_int4(
    x: torch.Tensor,
    weight_int4pack: torch.Tensor,
    scales_and_zeros: torch.Tensor,
    out_features: int,
    groupsize: int,
    precision: torch.dtype = ...,
    scales_precision: torch.dtype = ...,
):  # -> Any:
    ...

class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ...
    in_features: int
    out_features: int
    weight: torch.Tensor
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=...,
        device=...,
        dtype=...,
        groupsize: int = ...,
        inner_k_tiles: int = ...,
        precision: torch.dtype = ...,
        scales_precision: torch.dtype = ...,
    ) -> None: ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...

def replace_linear_int4(
    module, groupsize, inner_k_tiles, padding_allowed, skip_layer_func=...
):  # -> None:
    ...

class Int4WeightOnlyQuantizer(Quantizer):
    def __init__(
        self,
        groupsize: int = ...,
        padding_allowed: bool = ...,
        inner_k_tiles: int | None = ...,
        device: torch.device = ...,
        precision: torch.dtype = ...,
    ) -> None: ...
    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module: ...

def linear_forward_8da4w(
    x, weight_int8, bias, scales, zeros, out_features, groupsize, output_precision
):  # -> Tensor:
    ...

class Int8DynActInt4WeightLinear(torch.nn.Module):
    __constants__ = ...
    in_features: int
    out_features: int
    weight: torch.Tensor
    bias: torch.Tensor
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=...,
        device=...,
        dtype=...,
        groupsize: int = ...,
        precision: torch.dtype = ...,
        scales_precision: torch.dtype = ...,
    ) -> None: ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...

def replace_linear_8da4w(
    module: torch.nn.Module,
    groupsize: int,
    padding_allowed: bool,
    precision: torch.dtype,
    scales_precision: torch.dtype,
):  # -> None:
    ...

class Int8DynActInt4WeightQuantizer(Quantizer):
    def __init__(
        self,
        groupsize: int = ...,
        padding_allowed: bool = ...,
        precision: torch.dtype = ...,
        scales_precision: torch.dtype = ...,
        device: torch.device = ...,
        mapping_type: MappingType = ...,
    ) -> None: ...
    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module: ...
