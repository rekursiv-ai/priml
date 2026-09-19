from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any, Generic, TypeVar

import dataclasses

from _typeshed import Incomplete
from jax._src import (
    config as config,
    core as core,
    custom_api_util as custom_api_util,
    dtypes as dtypes,
    effects as effects,
    linear_util as lu,
    traceback_util as traceback_util,
)
from jax._src.ad_util import (
    SymbolicZero as SymbolicZero,
    Zero as Zero,
    p2tz as p2tz,
    stop_gradient_p as stop_gradient_p,
    zeros_like_aval as zeros_like_aval,
)
from jax._src.api_util import (
    argnums_partial as argnums_partial,
    debug_info as debug_info,
    flatten_fun_nokwargs as flatten_fun_nokwargs,
    fun_signature as fun_signature,
    infer_argnums_and_argnames as infer_argnums_and_argnames,
    prepend_static_args as prepend_static_args,
    resolve_kwargs as resolve_kwargs,
)
from jax._src.custom_transpose import custom_transpose as custom_transpose
from jax._src.errors import UnexpectedTracerError as UnexpectedTracerError
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
    pxla as pxla,
)
from jax._src.interpreters.batching import not_mapped as not_mapped
from jax._src.state.types import AbstractRef as AbstractRef
from jax._src.tree_util import (
    PyTreeDef as PyTreeDef,
    keystr as keystr,
    register_pytree_node_class as register_pytree_node_class,
    tree_flatten as tree_flatten,
    tree_flatten_with_path as tree_flatten_with_path,
    tree_leaves as tree_leaves,
    tree_leaves_with_path as tree_leaves_with_path,
    tree_map as tree_map,
    tree_structure as tree_structure,
    tree_unflatten as tree_unflatten,
    treedef_children as treedef_children,
    treedef_is_leaf as treedef_is_leaf,
    treedef_tuple as treedef_tuple,
)
from jax._src.util import (
    cache as cache,
    safe_map as safe_map,
    safe_zip as safe_zip,
    split_list as split_list,
    unzip2 as unzip2,
    weakref_lru_cache as weakref_lru_cache,
)

map = safe_map
zip = safe_zip
ReturnValue = TypeVar("ReturnValue")

class custom_jvp(Generic[ReturnValue]):
    fun: Callable[..., ReturnValue]
    nondiff_argnums: Sequence[int]
    nondiff_argnames: Sequence[str]
    jvp: Callable[..., tuple[ReturnValue, ReturnValue]] | None
    symbolic_zeros: bool
    def __init__(
        self,
        fun: Callable[..., ReturnValue],
        nondiff_argnums: Sequence[int] = (),
        nondiff_argnames: Sequence[str] = (),
    ) -> None: ...
    __getattr__: Incomplete
    def defjvp(
        self,
        jvp: Callable[..., tuple[ReturnValue, ReturnValue]],
        symbolic_zeros: bool = False,
    ) -> Callable[..., tuple[ReturnValue, ReturnValue]]: ...
    def defjvps(self, *jvps: Callable[..., ReturnValue] | None) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> ReturnValue: ...

class CustomJVPCallPrimitive(core.Primitive):
    multiple_results: bool
    def bind(self, *args, **params): ...
    def bind_with_trace(self, trace, args, params): ...
    def impl(self, fun, _, *args) -> None: ...
    def get_bind_params(self, params): ...

def lift_jvp(num_consts: int, jvp_jaxpr_fun: lu.WrappedFun) -> lu.WrappedFun: ...

custom_jvp_call_p: Incomplete

class custom_vjp(Generic[ReturnValue]):
    def __new__(cls, fun, nondiff_argnums=(), nondiff_argnames=()): ...
    fun: Incomplete
    nondiff_argnums: Incomplete
    fwd: Callable[..., tuple[ReturnValue, Any]] | None
    bwd: Callable[..., tuple[Any, ...]] | None
    symbolic_zeros: bool
    optimize_remat: bool
    def __init__(
        self,
        fun: Callable[..., ReturnValue],
        nondiff_argnums: Sequence[int] = (),
        nondiff_argnames: Sequence[str] = (),
    ) -> None: ...
    __getattr__: Incomplete
    def defvjp(
        self,
        fwd: Callable[..., tuple[ReturnValue, Any]],
        bwd: Callable[..., tuple[Any, ...]],
        symbolic_zeros: bool = False,
        optimize_remat: bool = False,
    ) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> ReturnValue: ...

@dataclasses.dataclass
class CustomVJPPrimal:
    value: Any
    perturbed: bool

def custom_vjp_primal_tree_values(tree): ...

class CustomVJPCallPrimitive(core.Primitive):
    multiple_results: bool
    def bind(self, *args, **params): ...
    def bind_with_trace(self, trace, args, params): ...
    def impl(self, fun, fwd, bwd, *args) -> None: ...
    def get_bind_params(self, params): ...

def lift_fwd(num_consts: int, fwd_jaxpr_thunk: lu.WrappedFun) -> lu.WrappedFun: ...

custom_vjp_call_p: Incomplete

def custom_gradient(fun): ...

class Residuals:
    jaxpr: Incomplete
    in_tree: Incomplete
    out_tree: Incomplete
    consts: Incomplete
    def __init__(self, jaxpr, in_tree, out_tree, consts) -> None: ...
    def __iter__(self): ...
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, aux, consts): ...

def closure_convert(fun: Callable, *example_args) -> tuple[Callable, list[Any]]: ...
def partition_list(choice, lst): ...
def linear_call(fun: Callable, fun_transpose: Callable, residual_args, linear_args): ...

linear_call_p: Incomplete
unreachable_p: core.Primitive

def unreachable_impl(*_, out_avals, exc_type, message) -> None: ...
def unreachable(*args, out_avals=None, exc_type=..., message: str = "unreachable"): ...

disallow_jvp: Incomplete

def custom_vjp_by_custom_transpose(fun, fwd, bwd): ...

custom_jvp_call_jaxpr_p: Incomplete

def optimize_remat_of_custom_vjp_fwd(
    fun: Callable[..., ReturnValue],
    debug_fun: core.DebugInfo,
    fwd: Callable[..., tuple[ReturnValue, Any]],
    debug_fwd: core.DebugInfo,
    nondiff_argnums: Sequence[int] = (),
    symbolic_zeros: bool = False,
) -> Callable[..., tuple[ReturnValue, Any]]: ...

remat_opt_p: Incomplete
