from typing import Any

from torchao.quantization.unified import TwoStepQuantizer

import torch

from .fake_quantize_config import FakeQuantizeConfigBase

class FakeQuantizedLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = ...,
        activation_config: FakeQuantizeConfigBase | None = ...,
        weight_config: FakeQuantizeConfigBase | None = ...,
        *args,
        **kwargs,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def to_linear(self) -> torch.nn.Linear: ...
    @classmethod
    def from_linear(
        cls,
        mod: torch.nn.Linear,
        activation_config: FakeQuantizeConfigBase | None = ...,
        weight_config: FakeQuantizeConfigBase | None = ...,
    ):  # -> FakeQuantizedLinear:
        ...

def enable_linear_fake_quant(mod: torch.nn.Module, enabled: bool = ...):  # -> None:
    ...
def disable_linear_fake_quant(mod: torch.nn.Module):  # -> None:
    ...

class _LegacyQATQuantizer(TwoStepQuantizer):
    def get_activation_fake_quantize_config(
        self,
    ) -> FakeQuantizeConfigBase | None: ...
    def get_weight_fake_quantize_config(self) -> FakeQuantizeConfigBase | None: ...

class Int8DynActInt4WeightQATQuantizer(_LegacyQATQuantizer):
    def __init__(
        self,
        groupsize: int = ...,
        padding_allowed: bool = ...,
        precision: torch.dtype = ...,
        scales_precision: torch.dtype = ...,
    ) -> None: ...
    def prepare(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module: ...
    def convert(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module: ...
    def get_activation_fake_quantize_config(
        self,
    ) -> FakeQuantizeConfigBase | None: ...
    def get_weight_fake_quantize_config(self) -> FakeQuantizeConfigBase | None: ...

class Int8DynActInt4WeightQATLinear(FakeQuantizedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = ...,
        device: torch.device = ...,
        groupsize: int = ...,
        precision: torch.dtype = ...,
        scales_precision: torch.dtype = ...,
    ) -> None: ...
    def enable_fake_quant(self, enabled: bool = ...):  # -> None:
        ...
    def disable_fake_quant(self):  # -> None:
        ...

def enable_8da4w_fake_quant(mod: torch.nn.Module):  # -> None:
    ...
def disable_8da4w_fake_quant(mod: torch.nn.Module):  # -> None:
    ...

class Int4WeightOnlyQATQuantizer(_LegacyQATQuantizer):
    def __init__(
        self,
        groupsize: int = ...,
        inner_k_tiles: int | None = ...,
        precision: torch.dtype = ...,
        scales_precision: torch.dtype = ...,
    ) -> None: ...
    def prepare(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module: ...
    def convert(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module: ...
    def get_weight_fake_quantize_config(self) -> FakeQuantizeConfigBase | None: ...

class Int4WeightOnlyQATLinear(FakeQuantizedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = ...,
        device: torch.device = ...,
        groupsize: int = ...,
        inner_k_tiles: int = ...,
        precision: torch.dtype = ...,
        scales_precision: torch.dtype = ...,
    ) -> None: ...
    def enable_fake_quant(self, enabled: bool = ...):  # -> None:
        ...
    def disable_fake_quant(self):  # -> None:
        ...

def enable_4w_fake_quant(mod: torch.nn.Module):  # -> None:
    ...
def disable_4w_fake_quant(mod: torch.nn.Module):  # -> None:
    ...

class Float8ActInt4WeightQATQuantizer(_LegacyQATQuantizer):
    def __init__(
        self, group_size: int | None = ..., scale_precision: torch.dtype = ...
    ) -> None: ...
    def prepare(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module: ...
    def convert(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module: ...
    def get_activation_fake_quantize_config(
        self,
    ) -> FakeQuantizeConfigBase | None: ...
    def get_weight_fake_quantize_config(self) -> FakeQuantizeConfigBase | None: ...
