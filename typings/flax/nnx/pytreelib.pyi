from abc import ABCMeta
from collections.abc import Generator

import dataclasses
import threading
import typing as tp

from _typeshed import Incomplete
from flax import (
    config as config,
    errors as errors,
    nnx as nnx,
)
from flax.nnx import (
    graphlib as graphlib,
    reprlib as reprlib,
    tracers as tracers,
    variablelib as variablelib,
    visualization as visualization,
)
from flax.nnx.variablelib import Variable as Variable
from flax.typing import (
    MISSING as MISSING,
    Missing as Missing,
    SizeBytes as SizeBytes,
)

import jax
import numpy as np

BUILDING_DOCS: Incomplete
A = tp.TypeVar("A")
P = tp.TypeVar("P", bound=Pytree)
T = tp.TypeVar("T", bound=type)
DataAnnotation: str
type Data[A] = tp.Annotated[A, DataAnnotation]
DATA_REGISTRY: set[type]

@tp.overload
def data(value: A, /) -> A: ...
@tp.overload
def data(
    *,
    default: A = ...,
    default_factory: tp.Callable[[], A] | None = None,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: tp.Mapping[str, tp.Any] | None = None,
    kw_only: bool = False,
) -> tp.Any: ...
def register_data_type(type_: T, /) -> T: ...
def is_data(value: tp.Any, /) -> bool: ...
def has_data(value: tp.Any, /) -> list[tp.Any]: ...

StaticAnnotation: str
type Static[A] = tp.Annotated[A, StaticAnnotation]

@tp.overload
def static(value: A, /) -> A: ...
@tp.overload
def static(
    *,
    default: A = ...,
    default_factory: tp.Callable[[], A] | None = None,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: tp.Mapping[str, tp.Any] | None = None,
    kw_only: bool = False,
) -> tp.Any: ...
@tp.overload
def dataclass(cls, /) -> type[A]: ...
@tp.overload
def dataclass(
    *,
    init: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
    weakref_slot: bool = False,
) -> tp.Callable[[type[A]], type[A]]: ...

@dataclasses.dataclass
class ObjectContext(threading.local):
    seen_modules_repr: set[int] | None = ...
    node_stats: dict[int, dict[type[Variable], SizeBytes]] | None = ...

OBJECT_CONTEXT: Incomplete

class PytreeState(reprlib.Representable):
    def __init__(self, initializing: bool = False, is_setup: bool = False) -> None: ...
    @property
    def trace_state(self) -> tracers.TraceState: ...
    @property
    def initializing(self) -> bool: ...
    @property
    def is_setup(self) -> bool: ...
    def __nnx_repr__(self) -> Generator[Incomplete]: ...
    def __treescope_repr__(self, path, subtree_renderer): ...

def check_pytree(pytree) -> None: ...

class PytreeMeta(ABCMeta): ...

ObjectMeta = PytreeMeta

@dataclasses.dataclass(frozen=True, repr=False)
class ArrayRepr(reprlib.Representable):
    shape: tuple[int, ...]
    dtype: tp.Any
    @staticmethod
    def from_array(array: jax.Array | np.ndarray) -> ArrayRepr: ...
    def __nnx_repr__(self) -> Generator[Incomplete]: ...

@dataclasses.dataclass(frozen=True, repr=False)
class VariableRepr(reprlib.Representable):
    var_type: type[Variable]
    value: tp.Any
    metadata: dict[str, tp.Any]
    def __nnx_repr__(self) -> Generator[Incomplete, Incomplete]: ...

@dataclasses.dataclass(frozen=True, repr=False)
class MutableArrayRepr(reprlib.Representable):
    shape: tuple[int, ...]
    dtype: tp.Any
    @staticmethod
    def from_array(array: jax.Array | np.ndarray) -> MutableArrayRepr: ...
    def __nnx_repr__(self) -> Generator[Incomplete]: ...

class AttributeStatus(tp.NamedTuple):
    is_data: bool
    explicit: bool

class Pytree(reprlib.Representable, metaclass=PytreeMeta):
    def __init_subclass__(cls, *, pytree: bool = ..., **kwargs) -> None: ...
    def __deepcopy__(self, memo=None) -> P: ...
    def __nnx_repr__(self) -> Generator[Incomplete]: ...
    def __treescope_repr__(self, path, subtree_renderer): ...
    def __delattr__(self, name: str) -> None: ...
    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any: ...

class Object(Pytree, pytree=False):
    def __init_subclass__(cls, **kwargs) -> None: ...
