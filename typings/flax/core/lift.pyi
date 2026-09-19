from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Generic, TypeVar

import contextlib
import dataclasses
import threading

from _typeshed import Incomplete
from flax import (
    traceback_util as traceback_util,
    traverse_util as traverse_util,
)
from flax.typing import (
    In as In,
    InOutAxis as InOutAxis,
    InOutScanAxis as InOutScanAxis,
    Out as Out,
)

from . import (
    axes_scan as axes_scan,
    meta as meta,
)
from .frozen_dict import (
    freeze as freeze,
    unfreeze as unfreeze,
)
from .scope import (
    CollectionFilter as CollectionFilter,
    DenyList as DenyList,
    Filter as Filter,
    LazyRng as LazyRng,
    PRNGSequenceFilter as PRNGSequenceFilter,
    Scope as Scope,
    group_collections as group_collections,
    in_filter as in_filter,
    intersect_filters as intersect_filters,
    is_filter_empty as is_filter_empty,
    subtract_filters as subtract_filters,
    union_filters as union_filters,
)

A = TypeVar("A")

@dataclasses.dataclass
class TransformContext(threading.local, Generic[A]):
    stack: list[A] = dataclasses.field(default_factory=list)
    @contextlib.contextmanager
    def push(self, a: A): ...
    def get(self) -> A: ...

def tree_map_rngs(fn, tree): ...
def pack(
    fn: Callable[..., Any],
    in_variable_filters: Sequence[CollectionFilter],
    out_variable_filters: Sequence[CollectionFilter],
    rng_filters: Sequence[PRNGSequenceFilter],
    name=None,
    enable_kwargs: bool = False,
) -> Callable[..., Any]: ...

id_fn: Incomplete

def map_variables(
    fn: Callable[..., Any],
    mapped_collections: CollectionFilter,
    map_in_fn: Callable[..., Any] = ...,
    map_out_fn: Callable[..., Any] = ...,
    init: bool = False,
    mutable: bool = False,
    rngs: PRNGSequenceFilter = True,
    variables: CollectionFilter = True,
) -> Callable[..., Any]: ...
def swap_collection(fn: Callable[..., Any], col_a: str, col_b: str): ...
def vjp(
    fn: Callable[..., Any],
    scope: Scope,
    *primals,
    has_aux: bool = False,
    reduce_axes=(),
    vjp_variables: CollectionFilter = "params",
    variables: CollectionFilter = True,
    rngs: PRNGSequenceFilter = True,
) -> tuple[Any, Callable[..., Any]] | tuple[Any, Callable[..., Any], Any]: ...
def value_and_grad(
    fn: Callable[..., Any],
    scope: Scope,
    *primals,
    has_aux: bool = False,
    reduce_axes=(),
    variables: CollectionFilter = True,
    rngs: PRNGSequenceFilter = True,
) -> tuple[Any, Callable[..., Any]] | tuple[Any, Callable[..., Any], Any]: ...
def jvp(
    fn: Callable[..., Any],
    scope: Scope,
    primals,
    tangents,
    variable_tangents,
    variables: CollectionFilter = True,
    rngs: PRNGSequenceFilter = True,
) -> tuple[Any, Any]: ...
def vmap(
    fn: Callable[..., Any],
    variable_axes: Mapping[CollectionFilter, InOutAxis],
    split_rngs: Mapping[PRNGSequenceFilter, bool],
    in_axes: int = 0,
    out_axes: int = 0,
    axis_size: int | None = None,
    axis_name: str | None = None,
    spmd_axis_name: str | None = None,
    metadata_params: dict[Any, Any] = {},
) -> Callable[..., Any]: ...
def scan(
    fn: Callable[..., Any],
    variable_axes: Mapping[CollectionFilter, InOutScanAxis] = {},
    variable_broadcast: CollectionFilter = False,
    variable_carry: CollectionFilter = False,
    split_rngs: Mapping[PRNGSequenceFilter, bool] = {},
    in_axes: int = 0,
    out_axes: int = 0,
    length: int | None = None,
    reverse: bool = False,
    unroll: int = 1,
    _split_transpose: bool = False,
    data_transform: Callable[..., Any] | None = None,
    metadata_params: dict[Any, Any] = {},
    check_constancy_invariants: bool = True,
) -> Callable[..., Any]: ...

C = TypeVar("C")

def while_loop(
    cond_fn: Callable[[Scope, C], bool],
    body_fn: Callable[[Scope, C], C],
    scope: Scope,
    init: C,
    carry_variables: CollectionFilter = False,
    broadcast_variables: CollectionFilter = True,
    split_rngs: Mapping[PRNGSequenceFilter, bool] = {},
) -> C: ...
def cond(
    pred: Any,
    true_fun: Callable[..., C],
    false_fun: Callable[..., C],
    scope: Scope,
    *operands,
    variables: CollectionFilter = True,
    rngs: PRNGSequenceFilter = True,
) -> C: ...
def switch(
    index: Any,
    branches: Sequence[Callable[..., C]],
    scope: Scope,
    *operands,
    variables: CollectionFilter = True,
    rngs: PRNGSequenceFilter = True,
) -> C: ...
def custom_vjp(
    fn: Callable[..., Any],
    forward_fn: Callable[..., Any],
    backward_fn: Callable[..., Any],
    grad_vars: CollectionFilter = "params",
    nondiff_argnums=(),
): ...
def checkpoint(
    fn: Callable[..., Any],
    variables: CollectionFilter = True,
    rngs: PRNGSequenceFilter = True,
    concrete: bool = False,
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: Callable[..., bool] | None = None,
) -> Callable[..., Any]: ...

remat = checkpoint

class CountsHolder:
    flat_d: Incomplete
    def __init__(self, flat_d) -> None: ...
    @classmethod
    def make(cls, d): ...
    def sub(self, other): ...
    def add(self, other): ...
    def unflat(self): ...

def set_from_dict(original, updates) -> None: ...

class _SideEffectCache(threading.local):
    cache: Incomplete
    def __init__(self) -> None: ...

def jit(
    fn: Callable[..., Any],
    variables: CollectionFilter = True,
    rngs: PRNGSequenceFilter = True,
    static_argnums: int | Iterable[int] = (),
    static_argnames: str | Iterable[str] = (),
    donate_argnums: int | Iterable[int] = (),
    device=None,
    backend: str | None = None,
) -> Callable[..., Any]: ...
def remat_scan(
    body_fn: Callable[..., Any],
    lengths: Sequence[int],
    policy: Callable[..., bool] | None = None,
    variable_broadcast: CollectionFilter = False,
    variable_carry: CollectionFilter = False,
    variable_axes: Mapping[CollectionFilter, InOutScanAxis] = {True: 0},
    split_rngs: Mapping[PRNGSequenceFilter, bool] = {True: True},
) -> Callable[..., Any]: ...
def fold_rngs(
    fn: Callable[..., Any],
    variables: CollectionFilter = True,
    rngs: PRNGSequenceFilter = True,
) -> Callable[..., Any]: ...
