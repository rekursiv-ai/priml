from collections.abc import Callable, Iterator, Mapping
from typing import Any, Literal, TypeVar, overload

import contextlib
import dataclasses
import enum
import threading
import typing_extensions as tpe
import weakref

from _typeshed import Incomplete
from flax import (
    config as config,
    core as core,
    errors as errors,
    serialization as serialization,
    traceback_util as traceback_util,
    traverse_util as traverse_util,
)
from flax.core import (
    Scope as Scope,
    meta as meta,
    partial_eval as partial_eval,
)
from flax.core.frozen_dict import FrozenDict as FrozenDict
from flax.core.scope import (
    CollectionFilter as CollectionFilter,
    DenyList as DenyList,
    Variable as Variable,
    union_filters as union_filters,
)
from flax.ids import (
    FlaxId as FlaxId,
    uuid as uuid,
)
from flax.linen import kw_only_dataclasses as kw_only_dataclasses
from flax.typing import (
    FrozenVariableDict as FrozenVariableDict,
    PRNGKey as PRNGKey,
    RNGSequences as RNGSequences,
    VariableDict as VariableDict,
)

T = TypeVar("T")
K = TypeVar("K")
M = TypeVar("M", bound=Module)
_CallableT = TypeVar("_CallableT", bound=Callable[..., Any])
TestScope: Incomplete

@dataclasses.dataclass
class _CallInfo:
    index: int
    path: tuple[str, ...]
    module: Module
    rngs: dict[str, core.scope.PRNGKey | core.scope.LazyRng] | None
    mutable: bool
    method: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    outputs: Any

@dataclasses.dataclass
class _CallInfoContext(threading.local):
    index: int
    calls: list[_CallInfo]
    def get_call_index(self) -> int: ...

class _DynamicContext(threading.local):
    module_stack: list[Module | None]
    capture_stack: Incomplete
    call_info_stack: list[_CallInfoContext]
    def __init__(self) -> None: ...

class _Sentinel:
    def __copy__(self): ...
    def __deepcopy__(self, memo): ...
    def __reduce__(self): ...

def enable_named_call() -> None: ...
def disable_named_call() -> None: ...
@contextlib.contextmanager
def override_named_call(enable: bool = True): ...

@dataclasses.dataclass(frozen=True)
class InterceptorContext:
    module: Module
    method_name: str
    orig_method: Callable[..., Any]

class ThreadLocalStack(threading.local):
    def __init__(self) -> None: ...
    def push(self, elem: Any) -> None: ...
    def pop(self) -> Any: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __len__(self) -> int: ...

type Args = tuple[Any]
type Kwargs = dict[str, Any]
type NextGetter = Callable[..., Any]
type Interceptor = Callable[[NextGetter, Args, Kwargs, InterceptorContext], Any]

@contextlib.contextmanager
def intercept_methods(interceptor: Interceptor): ...
def run_interceptors(
    orig_method: Callable[..., Any],
    module: Module,
    *args,
    **kwargs,
) -> Any: ...
def compact(fun: _CallableT) -> _CallableT: ...
def nowrap(fun: _CallableT) -> _CallableT: ...
def compact_name_scope(fun: _CallableT) -> _CallableT: ...
def wrap_method_once(fun: Callable[..., Any]) -> Callable[..., Any]: ...
def wrap_descriptor_once(descriptor) -> DescriptorWrapper: ...

class SetupState(enum.IntEnum):
    NEW = 0
    TRANSFORMED = 1
    DONE = 2

@dataclasses.dataclass
class _ModuleInternalState:
    in_compact_method: bool = ...
    in_setup: bool = ...
    setup_called: SetupState = ...
    is_initialized: bool = ...
    autoname_cursor: dict[str, int] = dataclasses.field(default_factory=dict)
    children: dict[str, str | Module] = dataclasses.field(default_factory=dict)
    def reset(self) -> None: ...
    def export(self) -> _ModuleInternalState: ...
    def reimport(self, other: _ModuleInternalState) -> None: ...

tuple_reduce: Incomplete
tuple_init: Incomplete
capture_call_intermediates: Incomplete

class ParentDescriptor:
    def __get__(self, obj, objtype=None): ...
    def __set__(self, obj, value) -> None: ...

class Descriptor(tpe.Protocol):
    __isabstractmethod__: bool
    def __get__(self, obj, objtype=None) -> Any: ...
    def __set__(self, obj, value) -> None: ...
    def __delete__(self, obj) -> None: ...
    def __set_name__(self, owner, name) -> None: ...

class DescriptorWrapper: ...

def create_descriptor_wrapper(descriptor: Descriptor): ...
def module_field(*, kw_only: bool = False, default: Any | None = ...) -> Any: ...

@tpe.dataclass_transform(field_specifiers=(module_field,))
class ModuleBase:
    scope: Scope | None
    __dataclass_fields__: dict[str, dataclasses.Field]

class Module(ModuleBase):
    name: str | None = module_field(kw_only=True, default=None)
    parent: Module | _Sentinel | None = module_field(kw_only=True, default=None)
    def __init__(self, *args, **kwargs) -> None: ...
    def __call__(self, *args, **kwargs) -> Any: ...
    @classmethod
    def __init_subclass__(cls, kw_only: bool = False, **kwargs: Any) -> None: ...
    def __setattr__(self, name: str, val: Any): ...
    def __getattr__(self, name: str) -> Any: ...
    def __dir__(self) -> list[str]: ...
    def __post_init__(self) -> None: ...
    def setup(self) -> None: ...
    @property
    def path(self): ...
    def clone(
        self,
        *,
        parent: Scope | Module | _Sentinel | None = None,
        _deep_clone: bool | weakref.WeakValueDictionary = False,
        _reset_names: bool = False,
        **updates,
    ) -> M: ...
    def copy(
        self,
        *,
        parent: Scope | Module | _Sentinel | None = ...,
        name: str | None = None,
        **updates,
    ) -> M: ...
    @overload
    def variable(
        self,
        col: str,
        name: str,
        init_fn: Callable[..., T] | None = None,
        *init_args,
    ) -> Variable[T]: ...
    @overload
    def variable(
        self,
        col: str,
        name: str,
        init_fn: Callable[..., T] | None = None,
        *init_args,
        unbox: Literal[True],
        **init_kwargs,
    ) -> Variable[T]: ...
    @overload
    def variable(
        self,
        col: str,
        name: str,
        init_fn: Callable[..., T] | None = None,
        *init_args,
        unbox: Literal[False],
        **init_kwargs,
    ) -> Variable[meta.AxisMetadata[T]]: ...
    @overload
    def variable(
        self,
        col: str,
        name: str,
        init_fn: Callable[..., T] | None = None,
        *init_args,
        unbox: bool = True,
        **init_kwargs,
    ) -> Variable[T] | Variable[meta.AxisMetadata[T]]: ...
    @overload
    def param(self, name: str, init_fn: Callable[..., T], *init_args) -> T: ...
    @overload
    def param(
        self,
        name: str,
        init_fn: Callable[..., meta.AxisMetadata[T]] | Callable[..., T],
        *init_args,
        unbox: Literal[True],
        **init_kwargs,
    ) -> T: ...
    @overload
    def param(
        self,
        name: str,
        init_fn: Callable[..., T],
        *init_args,
        unbox: Literal[False],
        **init_kwargs,
    ) -> T: ...
    @overload
    def param(
        self,
        name: str,
        init_fn: Callable[..., T | meta.AxisMetadata[T]],
        *init_args,
        unbox: bool,
        **init_kwargs,
    ) -> T | meta.AxisMetadata[T]: ...
    def has_variable(self, col: str, name: str) -> bool: ...
    def is_mutable_collection(self, col: str) -> bool: ...
    def has_rng(self, name: str) -> bool: ...
    def make_rng(self, name: str = "params") -> PRNGKey: ...
    def is_initializing(self) -> bool: ...
    @traceback_util.api_boundary
    def bind(
        self,
        variables: VariableDict,
        *args,
        rngs: RNGSequences | None = None,
        mutable: CollectionFilter = False,
    ) -> M: ...
    def unbind(self) -> tuple[M, VariableDict]: ...
    @traceback_util.api_boundary
    def apply(
        self,
        variables: VariableDict,
        *args,
        rngs: PRNGKey | RNGSequences | None = None,
        method: Callable[..., Any] | str | None = None,
        mutable: CollectionFilter = False,
        capture_intermediates: bool | Callable[[Module, str], bool] = False,
        **kwargs,
    ) -> tuple[Any, FrozenVariableDict | dict[str, Any]]: ...
    @traceback_util.api_boundary
    def init_with_output(
        self,
        rngs: PRNGKey | RNGSequences,
        *args,
        method: Callable[..., Any] | str | None = None,
        mutable: CollectionFilter = ...,
        capture_intermediates: bool | Callable[[Module, str], bool] = False,
        **kwargs,
    ) -> tuple[Any, FrozenVariableDict | dict[str, Any]]: ...
    @traceback_util.api_boundary
    def init(
        self,
        rngs: PRNGKey | RNGSequences,
        *args,
        method: Callable[..., Any] | str | None = None,
        mutable: CollectionFilter = ...,
        capture_intermediates: bool | Callable[[Module, str], bool] = False,
        **kwargs,
    ) -> FrozenVariableDict | dict[str, Any]: ...
    @traceback_util.api_boundary
    def lazy_init(
        self,
        rngs: PRNGKey | RNGSequences,
        *args,
        method: Callable[..., Any] | None = None,
        mutable: CollectionFilter = ...,
        **kwargs,
    ) -> FrozenVariableDict: ...
    @property
    def variables(self) -> VariableDict: ...
    def get_variable(self, col: str, name: str, default: T | None = None) -> T: ...
    def put_variable(self, col: str, name: str, value: Any): ...
    @overload
    def sow(self, col: str, name: str, value: Any) -> bool: ...
    @overload
    def sow(
        self,
        col: str,
        name: str,
        value: T,
        reduce_fn: Callable[[K, T], K] = ...,
        init_fn: Callable[[], K] = ...,
    ) -> bool: ...
    def perturb(self, name: str, value: T, collection: str = "perturbations") -> T: ...
    def tabulate(
        self,
        rngs: PRNGKey | RNGSequences,
        *args,
        depth: int | None = None,
        show_repeated: bool = False,
        mutable: CollectionFilter = ...,
        console_kwargs: Mapping[str, Any] | None = None,
        table_kwargs: Mapping[str, Any] = ...,
        column_kwargs: Mapping[str, Any] = ...,
        compute_flops: bool = False,
        compute_vjp_flops: bool = False,
        **kwargs,
    ) -> str: ...
    def module_paths(
        self,
        rngs: PRNGKey | RNGSequences,
        *args,
        show_repeated: bool = False,
        mutable: CollectionFilter = ...,
        **kwargs,
    ) -> dict[str, Module]: ...

def merge_param(name: str, a: T | None, b: T | None) -> T: ...
@traceback_util.api_boundary
def apply(
    fn: Callable[..., Any],
    module: Module,
    mutable: CollectionFilter = False,
    capture_intermediates: bool | Callable[[Module, str], bool] = False,
) -> Callable[..., Any]: ...
@traceback_util.api_boundary
def init_with_output(
    fn: Callable[..., Any],
    module: Module,
    mutable: CollectionFilter = ...,
    capture_intermediates: bool | Callable[[Module, str], bool] = False,
) -> Callable[..., tuple[Any, FrozenVariableDict | dict[str, Any]]]: ...
@traceback_util.api_boundary
def init(
    fn: Callable[..., Any],
    module: Module,
    mutable: CollectionFilter = ...,
    capture_intermediates: bool | Callable[[Module, str], bool] = False,
) -> Callable[..., FrozenVariableDict | dict[str, Any]]: ...

@dataclasses.dataclass
class CompactNameScope:
    fn: Callable
    module_fn: Callable
    name: str
    def __call__(self, *args, **kwargs) -> Any: ...

def share_scope(module: Module, other: Module, /): ...
