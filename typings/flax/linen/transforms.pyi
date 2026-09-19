from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TypeVar

import contextlib
import dataclasses
import weakref

from flax import (
    core as core,
    errors as errors,
    serialization as serialization,
    struct as struct,
    traceback_util as traceback_util,
)
from flax.core import (
    Scope as Scope,
    lift as lift,
    meta as meta,
)
from flax.core.frozen_dict import FrozenDict as FrozenDict
from flax.core.scope import (
    CollectionFilter as CollectionFilter,
    LazyRng as LazyRng,
    PRNGSequenceFilter as PRNGSequenceFilter,
)
from flax.ids import FlaxId as FlaxId
from flax.linen.module import (
    Module as Module,
    Variable as Variable,
    wrap_method_once as wrap_method_once,
)
from flax.typing import (
    InOutAxis as InOutAxis,
    InOutScanAxis as InOutScanAxis,
)

def clean_clone(x): ...

@struct.dataclass
class VariablePlaceholder:
    collection: str = struct.field(pytree_node=False)
    name: str = struct.field(pytree_node=False)
    unbox: bool = struct.field(pytree_node=False)
    id: int = struct.field(pytree_node=False)

@struct.dataclass
class InstancePlaceholder:
    cls: type[Any] = struct.field(pytree_node=False)
    attrs: dict[Any, Any] = struct.field(pytree_node=False)
    id: int = struct.field(pytree_node=False)

def get_module_scopes(module, args=None, kwargs=None): ...
def set_module_scopes(module, args, kwargs, scopes): ...
def module_class_lift_transform(
    transform,
    module_class,
    *trafo_args,
    methods=None,
    **trafo_kwargs,
): ...
def decorator_lift_transform(
    transform,
    class_fn,
    *trafo_args,
    multi_scope: bool = True,
    **trafo_kwargs,
): ...

@dataclasses.dataclass(frozen=True)
class _HashableProxy:
    module_ref: weakref.ref
    hash_key: int
    @classmethod
    def from_module(cls, module: Module) -> _HashableProxy: ...
    def __hash__(self): ...
    def __eq__(self, other): ...
    @property
    def module(self): ...

def decorator_lift_transform_cached(transform, class_fn, **trafo_kwargs): ...
@contextlib.contextmanager
def fork_rngs(module: Module): ...
def module_class_lift_transform_cached(
    transform,
    module_class,
    methods=None,
    **trafo_kwargs,
): ...

type TransformTarget = type[Module] | Callable[..., Any]
Target = TypeVar("Target", bound=TransformTarget)

def lift_transform(transform, target, *trafo_args, methods=None, **trafo_kwargs): ...
def lift_transform_cached(
    transform,
    target,
    *trafo_args,
    methods=None,
    **trafo_kwargs,
): ...
def lift_direct_transform(
    transform: Callable[..., Any],
    targets: tuple[Callable[..., Any], ...],
    mdl: Module,
    *args,
    multi_scope: bool = True,
    **kwargs,
): ...
def vmap(
    target: Target,
    variable_axes: Mapping[CollectionFilter, InOutAxis] = ...,
    split_rngs: Mapping[PRNGSequenceFilter, bool] = ...,
    in_axes: int = 0,
    out_axes: int = 0,
    axis_size: int | None = None,
    axis_name: str | None = None,
    spmd_axis_name: str | None = None,
    metadata_params: Mapping[Any, Any] = {},
    methods=None,
) -> Target: ...
def jit(
    target: Target,
    variables: CollectionFilter = True,
    rngs: PRNGSequenceFilter = True,
    static_argnums: int | Iterable[int] = (),
    static_argnames: str | Iterable[str] = (),
    donate_argnums: int | Iterable[int] = (),
    device=None,
    backend: str | None = None,
    methods=None,
) -> Target: ...
def checkpoint(
    target: Target,
    variables: CollectionFilter = True,
    rngs: PRNGSequenceFilter = True,
    concrete: bool = False,
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: Callable[..., bool] | None = None,
    methods=None,
) -> Target: ...

remat = checkpoint

def remat_scan(
    target: Target,
    lengths: Sequence[int] | None = (),
    policy: Callable[..., bool] | None = None,
    variable_broadcast: CollectionFilter = False,
    variable_carry: CollectionFilter = False,
    variable_axes: Mapping[CollectionFilter, InOutScanAxis] = ...,
    split_rngs: Mapping[PRNGSequenceFilter, bool] = ...,
) -> Target: ...
def scan(
    target: Target,
    variable_axes: Mapping[CollectionFilter, InOutScanAxis] = ...,
    variable_broadcast: CollectionFilter = False,
    variable_carry: CollectionFilter = False,
    split_rngs: Mapping[PRNGSequenceFilter, bool] = ...,
    in_axes: int = 0,
    out_axes: int = 0,
    length: int | None = None,
    reverse: bool = False,
    unroll: int = 1,
    data_transform: Callable[..., Any] | None = None,
    metadata_params: Mapping[Any, Any] = {},
    methods=None,
    _split_transpose: bool = False,
    check_constancy_invariants: bool = True,
) -> Target: ...
def map_variables(
    target: Target,
    mapped_collections: CollectionFilter = True,
    trans_in_fn: Callable[..., Any] = ...,
    trans_out_fn: Callable[..., Any] = ...,
    init: bool = False,
    mutable: bool = False,
    rngs: PRNGSequenceFilter = True,
    variables: CollectionFilter = True,
    methods=None,
) -> Target: ...
def vjp(
    fn: Callable[..., Any],
    mdl: Module,
    *primals,
    has_aux: bool = False,
    reduce_axes=(),
    vjp_variables: CollectionFilter = "params",
    variables: CollectionFilter = True,
    rngs: PRNGSequenceFilter = True,
    multi_scope: bool = False,
): ...
def value_and_grad(
    fn: Callable[..., Any],
    mdl: Module,
    *primals,
    has_aux: bool = False,
    reduce_axes=(),
    variables: CollectionFilter = True,
    rngs: PRNGSequenceFilter = True,
): ...
def grad(
    fn: Callable[..., Any],
    mdl: Module,
    *primals,
    has_aux: bool = False,
    reduce_axes=(),
    variables: CollectionFilter = True,
    rngs: PRNGSequenceFilter = True,
): ...
def jvp(
    fn: Callable[..., Any],
    mdl: Module,
    primals,
    tangents,
    variable_tangents,
    variables: CollectionFilter = True,
    rngs: PRNGSequenceFilter = True,
) -> tuple[Any, Callable[..., Any]] | tuple[Any, Callable[..., Any], Any]: ...

ModuleT = TypeVar("ModuleT", bound=Module)
C = TypeVar("C")

def while_loop(
    cond_fn: Callable[[ModuleT, C], bool],
    body_fn: Callable[[ModuleT, C], C],
    mdl: ModuleT,
    init: C,
    carry_variables: CollectionFilter = False,
    broadcast_variables: CollectionFilter = True,
    split_rngs: Mapping[PRNGSequenceFilter, bool] = ...,
) -> C: ...
def cond(
    pred: Any,
    true_fun: Callable[..., C],
    false_fun: Callable[..., C],
    mdl: Module,
    *operands,
    variables: CollectionFilter = True,
    rngs: PRNGSequenceFilter = True,
) -> C: ...
def switch(
    index: Any,
    branches: Sequence[Callable[..., C]],
    mdl: Module,
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
def named_call(class_fn, force: bool = True): ...
def add_metadata_axis(
    target: Target,
    variable_axes: Mapping[CollectionFilter, InOutAxis] = ...,
    metadata_params: dict[Any, Any] = {},
) -> Target: ...
def fold_rngs(
    target: Target,
    variables: CollectionFilter = True,
    rngs: PRNGSequenceFilter = True,
) -> Target: ...
