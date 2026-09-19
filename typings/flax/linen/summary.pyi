from abc import ABC, abstractmethod
from collections.abc import (
    Callable as Callable,
    Iterable,
    Mapping,
    Sequence,
)
from typing import Any

import abc
import dataclasses

from _typeshed import Incomplete
from flax.core import (
    meta as meta,
    unfreeze as unfreeze,
)
from flax.core.scope import (
    CollectionFilter as CollectionFilter,
    DenyList as DenyList,
    LazyRng as LazyRng,
)
from flax.typing import (
    Array as Array,
    FrozenVariableDict as FrozenVariableDict,
    LogicalNames as LogicalNames,
    MutableVariableDict as MutableVariableDict,
    PRNGKey as PRNGKey,
    RNGSequences as RNGSequences,
)

import flax.linen.module as module_lib

class _ValueRepresentation(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def render(self) -> str: ...

@dataclasses.dataclass
class _ArrayRepresentation(_ValueRepresentation):
    shape: tuple[int, ...]
    dtype: Any
    @classmethod
    def from_array(cls, x: Array) -> _ArrayRepresentation: ...
    @classmethod
    def render_array(cls, x) -> str: ...
    def render(self): ...

@dataclasses.dataclass
class _PartitionedArrayRepresentation(_ValueRepresentation):
    array_representation: _ArrayRepresentation
    names: LogicalNames
    @classmethod
    def from_partitioned(
        cls,
        partitioned: meta.Partitioned,
    ) -> _PartitionedArrayRepresentation: ...
    def render(self): ...

@dataclasses.dataclass
class _ObjectRepresentation(_ValueRepresentation):
    obj: Any
    def render(self): ...

@dataclasses.dataclass
class Row:
    path: tuple[str, ...]
    module_copy: module_lib.Module
    method: str
    inputs: Any
    outputs: Any
    module_variables: dict[str, dict[str, Any]]
    counted_variables: dict[str, dict[str, Any]]
    flops: int
    vjp_flops: int
    def __post_init__(self) -> None: ...
    def size_and_bytes(
        self,
        collections: Iterable[str],
    ) -> dict[str, tuple[int, int]]: ...

class Table(list[Row]):
    module: Incomplete
    collections: Incomplete
    def __init__(
        self,
        module: module_lib.Module,
        collections: Sequence[str],
        rows: Iterable[Row],
    ) -> None: ...

def tabulate(
    module: module_lib.Module,
    rngs: PRNGKey | RNGSequences,
    depth: int | None = None,
    show_repeated: bool = False,
    mutable: CollectionFilter = ...,
    console_kwargs: Mapping[str, Any] | None = None,
    table_kwargs: Mapping[str, Any] = ...,
    column_kwargs: Mapping[str, Any] = ...,
    compute_flops: bool = False,
    compute_vjp_flops: bool = False,
    **kwargs,
) -> Callable[..., str]: ...
