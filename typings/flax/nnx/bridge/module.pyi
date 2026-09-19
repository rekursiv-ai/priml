from collections import defaultdict

import dataclasses
import enum
import threading
import typing as tp
import typing_extensions as tpe

from _typeshed import Incomplete
from flax import errors as errors
from flax.core import meta as meta
from flax.core.frozen_dict import FrozenDict as FrozenDict
from flax.core.scope import CollectionFilter as CollectionFilter
from flax.nnx import (
    graphlib as graphlib,
    rnglib as rnglib,
    statelib as statelib,
    traversals as traversals,
    variablelib as variablelib,
)
from flax.nnx.pytreelib import (
    Pytree as Pytree,
    register_data_type as register_data_type,
)

import flax.nnx.module as nnx_module
import jax

A = tp.TypeVar("A")
M = tp.TypeVar("M", bound=Module)
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])

@dataclasses.dataclass
class ModuleStackEntry:
    module: Module
    in_compact: bool
    type_counter: defaultdict[type, int] = dataclasses.field(default_factory=Incomplete)

@dataclasses.dataclass
class ModuleContext(threading.local):
    module_stack: list[ModuleStackEntry | None] = dataclasses.field(
        default_factory=Incomplete,
    )

MODULE_CONTEXT: Incomplete

class ModuleState(statelib.State): ...

class Scope(Pytree):
    rngs: Incomplete
    mutable: Incomplete
    def __init__(self, rngs: rnglib.Rngs, mutable: CollectionFilter) -> None: ...
    def copy(self): ...

class _HasSetup(tp.Protocol):
    def setup(self) -> None: ...

def has_setup(x: tp.Any) -> tp.TypeGuard[_HasSetup]: ...
def current_context() -> ModuleStackEntry | None: ...
def current_module() -> Module | None: ...

class ModuleMeta(nnx_module.ModuleMeta): ...

class AttrPriority(enum.IntEnum):
    HIGH = 0
    INIT_PARENT = 20
    DEFAULT = 50
    LOW = 100

class PriorityStr(str):
    def __new__(cls, priority: AttrPriority, value: str): ...
    def __lt__(self, other) -> bool: ...
    def __gt__(self, other) -> bool: ...

class ModuleBase:
    scope: Scope | None
    attr_priorities: dict[str, AttrPriority]

@tpe.dataclass_transform(field_specifiers=(dataclasses.field,))
class Module(nnx_module.Module, ModuleBase, metaclass=ModuleMeta):
    def __init_subclass__(cls) -> None: ...
    def __getattribute__(self, name: str): ...
    def set_attr_priority(self, name: str, value: AttrPriority): ...
    def make_rng(self, name: str = "default") -> jax.Array: ...
    def param(
        self,
        name: str,
        init_fn: tp.Callable[..., A],
        *init_args,
        unbox: bool = True,
        **init_kwargs,
    ) -> variablelib.Param[A]: ...
    def variable(
        self,
        col: str,
        name: str,
        init_fn: tp.Callable[..., A] | None = None,
        *init_args,
        unbox: bool = True,
        **init_kwargs,
    ) -> variablelib.Variable[A]: ...
    @property
    def variables(self): ...
    def apply(
        self,
        variables: dict[str, tp.Mapping],
        *args,
        rngs: int | jax.Array | dict[str, jax.Array] | rnglib.Rngs | None = None,
        method: tp.Callable[..., tp.Any] | str = "__call__",
        mutable: CollectionFilter = False,
        _initialize: bool = False,
        **kwargs,
    ) -> tp.Any: ...
    def init(
        self,
        rngs: int | jax.Array | dict[str, jax.Array] | rnglib.Rngs | None = None,
        *args,
        method: tp.Callable[..., tp.Any] | str = "__call__",
        **kwargs,
    ): ...
    def init_with_output(
        self,
        rngs: int | jax.Array | dict[str, jax.Array] | rnglib.Rngs | None = None,
        *args,
        method: tp.Callable[..., tp.Any] | str = "__call__",
        mutable: tp.Any = False,
        **kwargs,
    ) -> tuple[tp.Any, dict[str, tp.Mapping]]: ...
    def is_initializing(self) -> bool: ...

def compact(f: F) -> F: ...
