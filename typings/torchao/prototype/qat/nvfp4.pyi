from typing import Any
from dataclasses import dataclass

from torchao.quantization.qat import FakeQuantizeConfigBase

import torch

@dataclass
class NVFP4FakeQuantizeConfig(FakeQuantizeConfigBase):
    use_per_tensor_scale: bool = ...
    use_swizzled_scales: bool = ...
    use_triton_kernel: bool = ...

class _NVFP4QuantizedForwardFakeQuantizedBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        _input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation_config: NVFP4FakeQuantizeConfig,
        weight_config: NVFP4FakeQuantizeConfig,
    ) -> torch.Tensor: ...
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor: ...

class NVFP4FakeQuantizedLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = ...,
        activation_config: NVFP4FakeQuantizeConfig | None = ...,
        weight_config: NVFP4FakeQuantizeConfig | None = ...,
        *args,
        **kwargs,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    @classmethod
    def from_linear(
        cls,
        mod: torch.nn.Linear,
        activation_config: NVFP4FakeQuantizeConfig | None = ...,
        weight_config: NVFP4FakeQuantizeConfig | None = ...,
    ):  # -> NVFP4FakeQuantizedLinear:
        ...
