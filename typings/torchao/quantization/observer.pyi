from abc import abstractmethod
from typing import Any

import torch

from .granularity import Granularity
from .quant_primitives import MappingType, ZeroPointDomain

logger = ...

class _PartialWrapper:
    def __init__(self, p) -> None: ...
    def __call__(self, *args, **keywords): ...
    def __repr__(self): ...
    def with_args(self, *args, **kwargs):  # -> _PartialWrapper:
        ...

ABC: Any = ...

class AffineQuantizedObserverBase(ABC, torch.nn.Module):
    with_args = ...
    def __init__(
        self,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        granularity: Granularity,
        quant_min: int | None = ...,
        quant_max: int | None = ...,
        eps: float | None = ...,
        scale_dtype: torch.dtype | None = ...,
        zero_point_dtype: torch.dtype | None = ...,
        preserve_zero: bool = ...,
        zero_point_domain: ZeroPointDomain = ...,
    ) -> None: ...
    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]: ...

class AffineQuantizedMinMaxObserver(AffineQuantizedObserverBase):
    def forward(self, input: torch.Tensor):  # -> Tensor:
        ...
    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]: ...

class AffineQuantizedFixedQParamObserver(AffineQuantizedObserverBase):
    def __init__(
        self,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        granularity: Granularity,
        quant_min: int | None = ...,
        quant_max: int | None = ...,
        eps: float | None = ...,
        scale_dtype: torch.dtype | None = ...,
        zero_point_dtype: torch.dtype | None = ...,
        preserve_zero: bool = ...,
        zero_point_domain: ZeroPointDomain = ...,
        scale: torch.Tensor | None = ...,
        zero_point: torch.Tensor | None = ...,
    ) -> None: ...
    def set_qparams(self, scale, zero_point=...):  # -> None:
        ...
    def forward(self, input):  # -> Tensor:
        ...
    def calculate_qparams(self):  # -> tuple[Any, Tensor | Any]:
        ...

class AffineQuantizedMSEObserver(AffineQuantizedObserverBase):
    def __init__(
        self,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        granularity: Granularity,
        quant_min: int | None = ...,
        quant_max: int | None = ...,
        eps: float | None = ...,
        scale_dtype: torch.dtype | None = ...,
        zero_point_dtype: torch.dtype | None = ...,
        preserve_zero: bool = ...,
        zero_point_domain: ZeroPointDomain = ...,
        steps: int = ...,
        run_once: bool = ...,
    ) -> None: ...
    def mse(self, pred, expect, block_size):  # -> Tensor:
        ...
    def loss_fn(self, x, new_min, new_max):  # -> Tensor:
        ...
    def line_search(self, input):  # -> tuple[Tensor, Tensor]:
        ...
    def forward(self, input):  # -> Tensor:
        ...
    def calculate_qparams(self):  # -> Tuple[Tensor, Tensor]:
        ...
