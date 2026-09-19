from collections.abc import Callable, Hashable, Iterable, Sequence
from contextlib import contextmanager
from typing import Any, Literal, NamedTuple, TypeVar, overload

import atexit
import dataclasses

from _typeshed import Incomplete
from jax._src import (
    api_util as api_util,
    array as array,
    basearray as basearray,
    config as config,
    core as core,
    dispatch as dispatch,
    distributed as distributed,
    dtypes as dtypes,
    linear_util as lu,
    pjit as pjit,
    sharding_impls as sharding_impls,
    sharding_specs as sharding_specs,
    source_info_util as source_info_util,
    stages as stages,
    traceback_util as traceback_util,
    tree_util as tree_util,
    util as util,
)
from jax._src.api_util import (
    apply_flat_fun_nokwargs as apply_flat_fun_nokwargs,
    argnums_partial as argnums_partial,
    check_callable as check_callable,
    debug_info as debug_info,
    donation_vector as donation_vector,
    flat_out_axes as flat_out_axes,
    flatten_axes as flatten_axes,
    flatten_fun as flatten_fun,
    flatten_fun_nokwargs as flatten_fun_nokwargs,
    flatten_fun_nokwargs2 as flatten_fun_nokwargs2,
    rebase_donate_argnums as rebase_donate_argnums,
)
from jax._src.core import (
    ShapedArray as ShapedArray,
    eval_jaxpr as eval_jaxpr,
    shaped_abstractify as shaped_abstractify,
    typeof as typeof,
)
from jax._src.dtypes import canonicalize_value as canonicalize_value
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    pxla as pxla,
)
from jax._src.layout import Format as Format
from jax._src.lib import (
    jax_jit as jax_jit,
    pmap_lib as pmap_lib,
    xla_client as xc,
)
from jax._src.mesh import (
    Mesh as Mesh,
    get_abstract_mesh as get_abstract_mesh,
    get_concrete_mesh as get_concrete_mesh,
)
from jax._src.sharding import Sharding as Sharding
from jax._src.sharding_impls import (
    NamedSharding as NamedSharding,
    PartitionSpec as P,
    PmapSharding as PmapSharding,
)
from jax._src.traceback_util import api_boundary as api_boundary
from jax._src.tree_util import (
    Partial as Partial,
    PyTreeDef as PyTreeDef,
    broadcast_prefix as broadcast_prefix,
    equality_errors_pytreedef as equality_errors_pytreedef,
    generate_key_paths as generate_key_paths,
    keystr as keystr,
    prefix_errors as prefix_errors,
    register_dataclass as register_dataclass,
    register_pytree_node as register_pytree_node,
    tree_flatten as tree_flatten,
    tree_flatten_with_path as tree_flatten_with_path,
    tree_leaves as tree_leaves,
    tree_map as tree_map,
    tree_structure as tree_structure,
    tree_transpose as tree_transpose,
    tree_unflatten as tree_unflatten,
)
from jax._src.util import (
    safe_map as safe_map,
    safe_zip as safe_zip,
    unzip2 as unzip2,
    wraps as wraps,
)

config_ext: Incomplete
AxisName = Hashable
Device: Incomplete
F = TypeVar("F", bound=Callable)
T = TypeVar("T")
U = TypeVar("U")
map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
ShapeDtypeStruct: Incomplete
float0: Incomplete

class NotSpecified: ...

@overload
def jit(
    fun: Callable,
    /,
    *,
    in_shardings: Any = ...,
    out_shardings: Any = ...,
    static_argnums: int | Sequence[int] | None = ...,
    static_argnames: str | Iterable[str] | None = ...,
    donate_argnums: int | Sequence[int] | None = ...,
    donate_argnames: str | Iterable[str] | None = ...,
    keep_unused: bool = ...,
    device: xc.Device | None = ...,
    backend: str | None = ...,
    inline: bool = ...,
    compiler_options: dict[str, Any] | None = ...,
) -> pjit.JitWrapped: ...
@overload
def jit(
    *,
    in_shardings: Any = ...,
    out_shardings: Any = ...,
    static_argnums: int | Sequence[int] | None = ...,
    static_argnames: str | Iterable[str] | None = ...,
    donate_argnums: int | Sequence[int] | None = ...,
    donate_argnames: str | Iterable[str] | None = ...,
    keep_unused: bool = ...,
    device: xc.Device | None = ...,
    backend: str | None = ...,
    inline: bool = ...,
    compiler_options: dict[str, Any] | None = ...,
) -> Callable[[Callable], pjit.JitWrapped]: ...
@contextmanager
def disable_jit(disable: bool = True): ...
def grad(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable: ...
def value_and_grad(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[..., tuple[Any, Any]]: ...
def fwd_and_bwd(
    fun: Callable,
    argnums: int | Sequence[int],
    has_aux: bool = False,
    jitted: bool = True,
) -> tuple[Callable, Callable]: ...
def jacfwd(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
) -> Callable: ...
def jacrev(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> Callable: ...
def jacobian(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> Callable: ...
def hessian(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
) -> Callable: ...
def vmap(
    fun: F,
    in_axes: int | Sequence[Any] | None = 0,
    out_axes: Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    sum_match: bool = False,
) -> F: ...
def pmap(
    fun: Callable,
    axis_name: AxisName | None = None,
    *,
    in_axes: int | Sequence[Any] | None = 0,
    out_axes: Any = 0,
    static_broadcasted_argnums: int | Iterable[int] = (),
    devices: Sequence[xc.Device] | None = None,
    backend: str | None = None,
    axis_size: int | None = None,
    donate_argnums: int | Iterable[int] = (),
    global_arg_shapes: tuple[tuple[int, ...], ...] | None = None,
) -> Any: ...

class PmapCallInfo(NamedTuple):
    flat_fun: lu.WrappedFun
    in_tree: PyTreeDef
    out_tree: Callable[[], PyTreeDef]
    flat_args: Sequence[Any]
    donated_invars: Sequence[bool]
    in_axes_flat: Sequence[int | None]
    local_axis_size: int
    out_axes_thunk: Callable
    devices: Sequence[xc.Device] | None
    global_axis_size: int
    is_explicit_global_axis_size: bool

class _PmapFastpathData(NamedTuple):
    version: int
    xla_executable: xc.LoadedExecutable
    in_handler: Any
    out_handler: Any
    out_pytree_def: Any
    input_devices: Sequence[xc.Device]
    input_indices: Sequence[sharding_specs.Index]
    input_array_shardings: Sequence[Any]
    out_avals: Sequence[Any]
    out_array_shardings: Sequence[Any]
    out_committed: Sequence[Any]

def jvp(fun: Callable, primals, tangents, has_aux: bool = False) -> tuple[Any, ...]: ...
@overload
def linearize(
    fun: Callable,
    *primals,
    has_aux: Literal[False] = False,
) -> tuple[Any, Callable]: ...
@overload
def linearize(
    fun: Callable,
    *primals,
    has_aux: Literal[True],
) -> tuple[Any, Callable, Any]: ...
@overload
def vjp(
    fun: Callable[..., T],
    *primals: Any,
    has_aux: Literal[False] = False,
    reduce_axes: Sequence[AxisName] = (),
) -> tuple[T, Callable]: ...
@overload
def vjp(
    fun: Callable[..., tuple[T, U]],
    *primals: Any,
    has_aux: Literal[True],
    reduce_axes: Sequence[AxisName] = (),
) -> tuple[T, Callable, U]: ...

@dataclasses.dataclass(frozen=True)
class RSpec:
    idx: int
    primal: bool

def tuptree_map(f, treedef, x): ...

@dataclasses.dataclass(frozen=True)
class NotNeeded: ...

@dataclasses.dataclass(frozen=True)
class GradValue: ...

@dataclasses.dataclass(frozen=True)
class GradRef: ...

@dataclasses.dataclass
class VJP:
    fun: Callable
    in_tree: PyTreeDef
    out_tree: PyTreeDef
    args_res: list[Any]
    opaque_residuals: list[Any]
    jaxpr = ...
    def __call__(self, out_ct, *extra_args): ...
    def with_refs(self, *maybe_ct_refs): ...
    __hash__ = ...
    __eq__ = ...

def linear_transpose(fun: Callable, *primals, reduce_axes=()) -> Callable: ...
@overload
def make_jaxpr(
    fun: Callable,
    static_argnums: int | Sequence[int] = (),
    axis_env: Sequence[tuple[AxisName, int]] | None = None,
    return_shape: Literal[False] = ...,
) -> Callable[..., core.ClosedJaxpr]: ...
@overload
def make_jaxpr(
    fun: Callable,
    static_argnums: int | Sequence[int] = (),
    axis_env: Sequence[tuple[AxisName, int]] | None = None,
    return_shape: Literal[True] = ...,
) -> Callable[..., tuple[core.ClosedJaxpr, Any]]: ...
def pspec_to_sharding(name, val): ...
def device_put(
    x,
    device: xc.Device | Sharding | P | Format | Any | None = None,
    *,
    src: xc.Device | Sharding | P | Format | Any | None = None,
    donate: bool | Any = False,
    may_alias: bool | Any | None = None,
): ...
def device_put_sharded(shards: Sequence[Any], devices: Sequence[xc.Device]): ...
def device_put_replicated(x: Any, devices: Sequence[xc.Device]): ...
def device_get(x: Any): ...
def eval_shape(
    fun: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> object: ...
def named_call(fun: F, *, name: str | None = None) -> F: ...
def named_scope(name: str) -> source_info_util.ExtendNameStackContextManager: ...
def effects_barrier() -> None: ...
def block_until_ready(x): ...
def copy_to_host_async(x): ...
def clear_backends() -> None: ...
@atexit.register
def clean_up() -> None: ...
def live_arrays(platform=None): ...
def clear_caches() -> None: ...
