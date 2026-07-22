import torch

from .fake_quantize_config import (
    FakeQuantizeConfigBase,
    Float8FakeQuantizeConfig,
    Int4WeightFakeQuantizeConfig,
    IntxFakeQuantizeConfig,
)

class FakeQuantizerBase(torch.nn.Module):
    config: FakeQuantizeConfigBase

    @staticmethod
    def from_config(config: FakeQuantizeConfigBase) -> FakeQuantizerBase: ...

class Float8FakeQuantizer(FakeQuantizerBase):
    def __init__(self, config: Float8FakeQuantizeConfig) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Int4WeightFakeQuantizer(FakeQuantizerBase):
    def __init__(self, config: Int4WeightFakeQuantizeConfig) -> None: ...
    def forward(self, w: torch.Tensor) -> torch.Tensor: ...

class IntxFakeQuantizer(FakeQuantizerBase):
    def __init__(self, config: IntxFakeQuantizeConfig) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class FakeQuantizer(IntxFakeQuantizer):
    def __init__(self, config: FakeQuantizeConfigBase) -> None: ...
