from typing import Any

from torch import nn
from torch.distributed.tensor import Placement
from torch.distributed.tensor.parallel import ParallelStyle
from transformers.processing_utils import Unpack

import torch

class GatedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: int | None = ...,
        intermediate_size: int | None = ...,
        hidden_act: str = ...,
        fuse_swiglu: bool = ...,
    ) -> GatedMLP: ...
    def forward(self, x: torch.Tensor, **kwargs: Unpack[Any]) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class SwiGLULinear(nn.Module):
    def forward(self, x, y, weight, bias) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...

class SwiGLULinearParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Placement | None = ...,
        output_layouts: Placement | None = ...,
        use_local_output: bool = ...,
    ) -> None: ...
