from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from torchao.core.config import AOBaseConfig
from torchao.quantization.unified import TwoStepQuantizer

import torch

from .fake_quantize_config import FakeQuantizeConfigBase

class QATStep(StrEnum):
    PREPARE = ...
    CONVERT = ...

@dataclass
class QATConfig(AOBaseConfig):
    base_config: AOBaseConfig | None
    activation_config: FakeQuantizeConfigBase | None
    weight_config: FakeQuantizeConfigBase | None
    step: QATStep
    def __init__(
        self,
        base_config: AOBaseConfig | None = ...,
        activation_config: FakeQuantizeConfigBase | None = ...,
        weight_config: FakeQuantizeConfigBase | None = ...,
        *,
        step: QATStep = ...,
    ) -> None: ...
    def __post_init__(self):  # -> None:
        ...

@dataclass
class IntXQuantizationAwareTrainingConfig(AOBaseConfig):
    activation_config: FakeQuantizeConfigBase | None = ...
    weight_config: FakeQuantizeConfigBase | None = ...
    def __post_init__(self):  # -> None:
        ...

class intx_quantization_aware_training(IntXQuantizationAwareTrainingConfig): ...

@dataclass
class FromIntXQuantizationAwareTrainingConfig(AOBaseConfig):
    def __post_init__(self):  # -> None:
        ...

class from_intx_quantization_aware_training(
    FromIntXQuantizationAwareTrainingConfig
): ...

class ComposableQATQuantizer(TwoStepQuantizer):
    def __init__(self, quantizers: list[TwoStepQuantizer]) -> None: ...
    def prepare(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module: ...
    def convert(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module: ...

def initialize_fake_quantizers(
    model: torch.nn.Module, example_inputs: tuple[Any, ...]
) -> None: ...
