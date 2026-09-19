from collections.abc import (
    Callable as Callable,
    Iterable,
    Sequence,
)
from typing import Any, NoReturn

import inspect

from _typeshed import Incomplete
from jax._src import (
    config as config,
    core as core,
    dtypes as dtypes,
    linear_util as lu,
    traceback_util as traceback_util,
)
from jax._src.state.types import AbstractRef as AbstractRef
from jax._src.tree_util import (
    PyTreeDef as PyTreeDef,
    broadcast_flattened_prefix_with_treedef as broadcast_flattened_prefix_with_treedef,
    broadcast_prefix as broadcast_prefix,
    generate_key_paths as generate_key_paths,
    none_leaf_registry as none_leaf_registry,
    prefix_errors as prefix_errors,
    tree_flatten as tree_flatten,
    tree_unflatten as tree_unflatten,
    treedef_children as treedef_children,
)
from jax._src.util import (
    HashableFunction as HashableFunction,
    Unhashable as Unhashable,
    safe_map as safe_map,
    safe_zip as safe_zip,
)

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete

@lu.transformation_with_aux2
def flatten_fun(f: Callable, store: lu.Store, in_tree: PyTreeDef, *args_flat): ...
def apply_flat_fun(fun, io_tree, *py_args): ...
@lu.transformation_with_aux2
def flatten_fun_nokwargs(
    f: Callable,
    store: lu.Store,
    in_tree: PyTreeDef,
    *args_flat,
): ...
def apply_flat_fun_nokwargs(fun, io_tree, py_args): ...
@lu.transformation_with_aux2
def flatten_fun_nokwargs2(f, store, in_tree, *args_flat): ...

class _HashableWithStrictTypeEquality:
    val: Incomplete
    def __init__(self, val) -> None: ...
    def __hash__(self): ...
    def __eq__(self, other): ...

def argnums_partial(
    f: lu.WrappedFun,
    dyn_argnums: int | Sequence[int],
    args: Sequence,
    require_static_args_hashable: bool = True,
): ...
def prepend_static_args(f, static_args): ...
def donation_vector(
    donate_argnums,
    donate_argnames,
    in_tree,
    kws: bool = True,
) -> tuple[bool, ...]: ...
def rebase_donate_argnums(donate_argnums, static_argnums) -> tuple[int, ...]: ...
def is_hashable(arg): ...

SENTINEL: Incomplete

def flatten_axes(
    name,
    treedef,
    axis_tree,
    *,
    kws: bool = False,
    tupled_args: bool = False,
): ...
def flat_out_axes(
    f: lu.WrappedFun,
    out_spec: Any,
) -> tuple[lu.WrappedFun, Callable]: ...
def check_callable(fun) -> None: ...
def infer_argnums_and_argnames(
    sig: inspect.Signature,
    argnums: int | Iterable[int] | None,
    argnames: str | Iterable[str] | None,
) -> tuple[tuple[int, ...], tuple[str, ...]]: ...
def resolve_argnums(
    fun: Callable,
    signature: inspect.Signature | None,
    donate_argnums: int | Sequence[int] | None,
    donate_argnames: str | Iterable[str] | None,
    static_argnums: int | Sequence[int] | None,
    static_argnames: str | Iterable[str] | None,
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[int, ...], tuple[str, ...]]: ...
def resolve_kwargs(fun: Callable, args, kwargs) -> tuple[Any, ...]: ...
def api_hook(fun, tag: str): ...
def debug_info(
    traced_for: str,
    fun: Callable,
    args: Sequence[Any],
    kwargs: dict[str, Any],
    *,
    static_argnums: Sequence[int] = (),
    static_argnames: Sequence[str] = (),
    result_paths_thunk: Callable[[], tuple[str, ...]] | core.InitialResultPaths = ...,
    sourceinfo: str | None = None,
    signature: inspect.Signature | None = None,
) -> core.DebugInfo: ...
def fun_signature(fun: Callable) -> inspect.Signature | None: ...
def save_wrapped_fun_debug_info(wrapper: Callable, dbg: core.DebugInfo) -> None: ...
def fun_sourceinfo(fun: Callable) -> str: ...

class _HashableByObjectId:
    val: Incomplete
    def __init__(self, val) -> None: ...
    def __hash__(self): ...
    def __eq__(self, other): ...

def check_no_aliased_ref_args(
    dbg_fn: Callable[[], core.DebugInfo],
    maybe_avals,
    args,
) -> None: ...
def check_no_transformed_refs_args(
    dbg_fn: Callable[[], core.DebugInfo],
    args_flat,
) -> None: ...

class InternalFloatingPointError(Exception):
    name: str
    ty: str
    def __init__(self, name: str, ty: str) -> None: ...

def maybe_recursive_nan_check(
    e: Exception,
    fun: Callable,
    args,
    kwargs,
) -> NoReturn: ...
