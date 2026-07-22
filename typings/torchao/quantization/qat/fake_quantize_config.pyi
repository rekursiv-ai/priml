from dataclasses import dataclass
from typing import Any

import abc

from torchao.float8.inference import FP8Granularity
from torchao.quantization.granularity import Granularity
from torchao.quantization.quant_primitives import (
    MappingType,
    TorchAODType,
    ZeroPointDomain,
)

import torch

class FakeQuantizeConfigBase(abc.ABC): ...

@dataclass
class Float8FakeQuantizeConfig(FakeQuantizeConfigBase):
    dtype: torch.dtype = ...
    granularity: FP8Granularity = ...
    hp_value_lb: float | None = ...
    hp_value_ub: float | None = ...
    def __post_init__(self):  # -> None:
        ...

@dataclass
class Int4WeightFakeQuantizeConfig(FakeQuantizeConfigBase):
    group_size: int = ...
    activation_dtype: torch.dtype = ...
    def __post_init__(self):  # -> None:
        ...

@dataclass
class IntxFakeQuantizeConfig(FakeQuantizeConfigBase):
    dtype: torch.dtype | TorchAODType
    granularity: Granularity
    mapping_type: MappingType
    scale_precision: torch.dtype
    zero_point_precision: torch.dtype
    zero_point_domain: ZeroPointDomain
    is_dynamic: bool = ...
    range_learning: bool = ...
    eps: float | None = ...
    def __init__(
        self,
        dtype: torch.dtype | TorchAODType,
        granularity: Granularity | str | None = ...,
        mapping_type: MappingType | None = ...,
        scale_precision: torch.dtype = ...,
        zero_point_precision: torch.dtype = ...,
        zero_point_domain: ZeroPointDomain = ...,
        is_dynamic: bool = ...,
        range_learning: bool = ...,
        eps: float | None = ...,
        *,
        group_size: int | None = ...,
        is_symmetric: bool | None = ...,
    ) -> None: ...
    def __post_init__(self):  # -> None:
        ...
    @property
    def group_size(self) -> int: ...
    @property
    def is_symmetric(self) -> bool: ...
    def __setattr__(self, name: str, value: Any):  # -> None:
        ...

class FakeQuantizeConfig(IntxFakeQuantizeConfig):
    def __post_init__(self):  # -> None:
        ...
