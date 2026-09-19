from collections.abc import (
    Callable as Callable,
    Generator,
    Iterable,
    Mapping,
    Sequence,
)
from typing import Any, Generic, Literal, TypeVar, overload

import contextlib
import dataclasses

from _typeshed import Incomplete
from flax import (
    errors as errors,
    struct as struct,
    traceback_util as traceback_util,
)
from flax.ids import uuid as uuid
from flax.typing import (
    Array as Array,
    Collection as Collection,
    MutableCollection as MutableCollection,
    MutableVariableDict as MutableVariableDict,
    PRNGFoldable as PRNGFoldable,
    PRNGKey as PRNGKey,
    RNGSequences as RNGSequences,
    VariableDict as VariableDict,
)

from . import (
    meta as meta,
    partial_eval as partial_eval,
    tracers as tracers,
)
from .frozen_dict import (
    FrozenDict as FrozenDict,
    freeze as freeze,
    unfreeze as unfreeze,
)

T = TypeVar("T")
Filter: Incomplete

@dataclasses.dataclass(frozen=True, eq=True)
class DenyList:
    deny: Filter

CollectionFilter = Filter
PRNGSequenceFilter = Filter

class LazyRng(struct.PyTreeNode):
    rng: PRNGKey
    suffix: tuple[PRNGFoldable, ...] = struct.field(pytree_node=False)
    def as_jax_rng(self) -> PRNGKey: ...
    @staticmethod
    def create(rng: LazyRng | PRNGKey, *suffix: PRNGFoldable) -> LazyRng: ...
    def clear_suffix(self): ...

def is_filter_empty(filter_like: Filter) -> bool: ...
def in_filter(filter_like: Filter, col: str) -> bool: ...
def filter_to_set(x: Filter) -> set[str]: ...
def union_filters(a: Filter, b: Filter) -> Filter: ...
def subtract_filters(a: Filter, b: Filter) -> Filter: ...
def intersect_filters(a: Filter, b: Filter) -> Filter: ...
def group_collections(
    xs: VariableDict,
    col_filters: Sequence[CollectionFilter],
) -> Sequence[MutableVariableDict]: ...

class Variable(Generic[T]):
    scope: Incomplete
    collection: Incomplete
    name: Incomplete
    unbox: Incomplete
    def __init__(
        self,
        scope: Scope,
        collection: str,
        name: str,
        unbox: bool,
    ) -> None: ...
    @property
    def value(self) -> T: ...
    @value.setter
    def value(self, value: T): ...
    def is_mutable(self) -> bool: ...

class _ChildRNGSentinel: ...

child_rng_token: Incomplete

class _DefaultSentinel: ...

no_flag: Incomplete

class Scope:
    reservations: dict[str, set[str | None]]
    parent: Incomplete
    name: Incomplete
    path: Incomplete
    debug_path: Incomplete
    rngs: Incomplete
    mutable: Incomplete
    flags: Incomplete
    trace_level: Incomplete
    rng_counters: Incomplete
    def __init__(
        self,
        variables: MutableVariableDict,
        rngs: RNGSequences | dict[str, LazyRng] | None = None,
        name: str | None = None,
        mutable: CollectionFilter = False,
        parent: Scope | None = None,
        path: Iterable[str] = (),
        debug_path: Iterable[str] = (),
        flags: Mapping | None = None,
    ) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    @property
    def root(self) -> Scope: ...
    @property
    def path_text(self) -> str: ...
    @property
    def invalid(self) -> bool: ...
    @contextlib.contextmanager
    def temporary(self) -> Generator[Incomplete]: ...
    def invalidate(self) -> None: ...
    def mutable_variables(self) -> VariableDict | dict[str, Any]: ...
    def variables(self) -> VariableDict | dict[str, Any]: ...
    def rewound(self, rewind_rngs: bool = False) -> Scope: ...
    def name_reserved(self, name: str, col: str | None = None) -> bool: ...
    def reserve(self, name: str, col: str | None = None): ...
    def default_name(self, prefix: str) -> str: ...
    def push(
        self,
        name: str | None = None,
        prefix: str = "",
        reuse: bool = False,
    ) -> Scope: ...
    def child(
        self,
        fn: Callable[..., Any],
        name: str | None = None,
        prefix: str | None = None,
        named_call: bool = True,
        **partial_kwargs,
    ) -> Callable[..., Any]: ...
    def is_mutable_collection(self, col: str) -> bool: ...
    def is_collection_empty(self, col: str) -> bool: ...
    def has_rng(self, name: str) -> bool: ...
    def make_rng(self, name: str = "params") -> PRNGKey: ...
    def get_variable(self, col: str, name: str, default: Any = None) -> Any: ...
    def has_variable(self, col: str, name: str) -> bool: ...
    def put_variable(self, col: str, name: str, value: Any): ...
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
    def has_flag(self, key) -> bool: ...
    def get_flag(self, key, default=...) -> Any: ...

def bind(
    variables: VariableDict,
    rngs: RNGSequences | None = None,
    mutable: CollectionFilter = False,
    flags: Mapping | None = None,
): ...
def apply(
    fn: Callable[..., Any],
    mutable: CollectionFilter = False,
    flags: Mapping | None = None,
) -> Callable[..., Any]: ...
def init(
    fn: Callable[..., Any],
    mutable: CollectionFilter = True,
    flags: Mapping | None = None,
) -> Callable[..., Any]: ...
def lazy_init(
    fn: Callable[..., Any],
    mutable: CollectionFilter = True,
    flags: Mapping | None = None,
) -> Callable[..., Any]: ...
