from collections import deque
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from typing import Any, Generic, Protocol, TypeGuard, TypeVar

import abc
import dataclasses
import types

from _typeshed import Incomplete
from flax.core import FrozenDict as FrozenDict

Array: Incomplete
PRNGKey: Incomplete
type RNGSequences = dict[str, PRNGKey]
Dtype: Incomplete
type Shape = Sequence[int]
K = TypeVar("K")

class Key(Hashable, Protocol):
    def __lt__(self, value: K) -> bool: ...

def is_key_like(x: Any) -> TypeGuard[Key]: ...

Path = str
type PathParts = tuple[Key, ...]
type Leaf = Any
PrecisionLike: Incomplete
type DotGeneralT = Callable[..., Array]
type ConvGeneralDilatedT = Callable[..., Array]
type EinsumT = Callable[..., Array]
type PaddingLike = str | int | Sequence[int | tuple[int, int]]
type LaxPadding = str | Sequence[tuple[int, int]]
Initializer: Incomplete
type Collection = Mapping[str, Any]
type MutableCollection = dict[str, Any]
type VariableDict = Mapping[str, Collection]
type FrozenVariableDict = FrozenDict[str, Collection]
type MutableVariableDict = dict[str, MutableCollection]
type PRNGFoldable = int | str
T = TypeVar("T")

@dataclasses.dataclass(frozen=True)
class In(Generic[T]):
    axis: T

@dataclasses.dataclass(frozen=True)
class Out(Generic[T]):
    axis: T

type Axis = int | None
type InOutAxis = Axis | In[Axis] | Out[Axis]
ScanAxis = int
type InOutScanAxis = ScanAxis | In[ScanAxis] | Out[ScanAxis]
type Axes = int | Sequence[int]
type LogicalNames = tuple[str | None, ...]
type AxisName = str | tuple[str, ...] | None
type LogicalRules = Sequence[tuple[str, AxisName]]
type ArrayPytree = Any
type LogicalPartitionSpec = Any
type LogicalPartitionSpecPytree = Any
type PartitionSpecPytree = Any
type Sharding = tuple[AxisName, ...]
A = TypeVar("A")
HA = TypeVar("HA", bound=Hashable)
HB = TypeVar("HB")

class PytreeDeque(deque[A]): ...
class Missing: ...

MISSING: Incomplete

class ShapeDtype(Protocol):
    shape: Shape
    dtype: Dtype

def has_shape_dtype(x: Any) -> TypeGuard[ShapeDtype]: ...

@dataclasses.dataclass(frozen=True, slots=True)
class SizeBytes:
    size: int
    bytes: int
    @classmethod
    def from_array(cls, x: ShapeDtype): ...
    def __add__(self, other: SizeBytes): ...
    def __bool__(self) -> bool: ...
    @classmethod
    def from_any(cls, x): ...

TupleArg = TypeVar("TupleArg", bound=tuple)

class PromoteDtypeFn(Protocol):
    def __call__(
        self,
        args: TupleArg,
        /,
        *,
        dtype: Any = None,
        inexact: bool = True,
    ) -> TupleArg: ...

class HashableMapping(Mapping[HA, HB], Hashable):
    def __init__(self, mapping: Mapping[HA, HB], copy: bool = True) -> None: ...
    def __contains__(self, key: object) -> bool: ...
    def __getitem__(self, key: HA) -> HB: ...
    def __iter__(self) -> Iterator[HA]: ...
    def __len__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def update(self, other: Mapping[HA, HB]) -> HashableMapping[HA, HB]: ...

F = TypeVar("F", bound=Callable[..., Any])

class BaseConfigContext(abc.ABC, metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def get_default(cls): ...
    @classmethod
    @abc.abstractmethod
    def get_stack(cls) -> list: ...
    prev_value: Incomplete
    new_value: Incomplete
    def __init__(self, value, /) -> None: ...
    @classmethod
    def current_value(cls): ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...
    def __call__(self, f: F) -> F: ...
