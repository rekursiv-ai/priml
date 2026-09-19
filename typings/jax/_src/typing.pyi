from collections.abc import Sequence
from types import EllipsisType
from typing import Any, Protocol

import enum

from jax._src.basearray import Array as Array

import numpy as np

DType = np.dtype
type ExtendedDType = Any

class SupportsDType(Protocol):
    @property
    def dtype(self, /) -> DType: ...

class SupportsShape(Protocol):
    @property
    def shape(self, /) -> tuple[int, ...]: ...

class SupportsSize(Protocol):
    @property
    def size(self, /) -> int: ...

class SupportsNdim(Protocol):
    @property
    def ndim(self, /) -> int: ...

type DTypeLike = str | type[Any] | np.dtype | SupportsDType
type DimSize = int | Any
type Shape = Sequence[DimSize]

class DuckTypedArray(Protocol):
    @property
    def dtype(self) -> DType: ...
    @property
    def shape(self) -> Shape: ...

class DeprecatedArg: ...

class DLDeviceType(enum.IntEnum):
    kDLCPU = 1
    kDLCUDA = 2
    kDLROCM = 10

type AnyInt = int | np.integer
type StaticIndex = AnyInt | slice | EllipsisType
type Index = StaticIndex | Sequence[AnyInt] | Array | np.ndarray | None
