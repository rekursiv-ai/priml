from collections.abc import (
    Callable as Callable,
    Iterable,
    Sequence,
)
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import abc
import inspect

from _typeshed import Incomplete
from jax._src import (
    api as api,
    api_util as api_util,
    config as config,
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
    effects as effects,
    op_shardings as op_shardings,
    profiler as profiler,
    sharding_impls as sharding_impls,
    source_info_util as source_info_util,
    stages as stages,
    traceback_util as traceback_util,
    tree_util as tree_util,
    util as util,
)
from jax._src.api_util import (
    check_callable as check_callable,
    check_no_aliased_ref_args as check_no_aliased_ref_args,
    debug_info as debug_info,
    donation_vector as donation_vector,
    flatten_axes as flatten_axes,
    resolve_argnums as resolve_argnums,
)
from jax._src.core import (
    cur_qdd as cur_qdd,
    typeof as typeof,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
    pxla as pxla,
    remat as remat,
)
from jax._src.layout import (
    AutoLayout as AutoLayout,
    Format as Format,
    Layout as Layout,
    get_layout_for_vmap as get_layout_for_vmap,
)
from jax._src.lib import (
    jax_jit as jax_jit,
    xla_client as xc,
)
from jax._src.lib.mlir import ir as ir
from jax._src.mesh import AbstractMesh as AbstractMesh
from jax._src.partition_spec import PartitionSpec as PartitionSpec
from jax._src.sharding import Sharding as Sharding
from jax._src.sharding_impls import (
    AUTO as AUTO,
    UNSPECIFIED as UNSPECIFIED,
    GSPMDSharding as GSPMDSharding,
    NamedSharding as NamedSharding,
    PmapSharding as PmapSharding,
    SingleDeviceSharding as SingleDeviceSharding,
    UnspecifiedValue as UnspecifiedValue,
    canonicalize_sharding as canonicalize_sharding,
    parse_flatten_op_sharding as parse_flatten_op_sharding,
    prepare_axis_resources as prepare_axis_resources,
)
from jax._src.state.types import (
    RefEffect as RefEffect,
    TransformedRefAvalError as TransformedRefAvalError,
)
from jax._src.traceback_util import api_boundary as api_boundary
from jax._src.tree_util import (
    FlatTree as FlatTree,
    PyTreeDef as PyTreeDef,
    prefix_errors as prefix_errors,
    tree_flatten as tree_flatten,
    tree_map as tree_map,
    tree_structure as tree_structure,
    tree_unflatten as tree_unflatten,
    treedef_children as treedef_children,
    treedef_is_leaf as treedef_is_leaf,
)
from jax._src.typing import ArrayLike as ArrayLike
from jax._src.util import (
    HashableFunction as HashableFunction,
    distributed_debug_log as distributed_debug_log,
    fun_name as fun_name,
    merge_lists as merge_lists,
    safe_map as safe_map,
    safe_zip as safe_zip,
    split_list as split_list,
    subs_list as subs_list,
    weakref_lru_cache as weakref_lru_cache,
    wraps as wraps,
)

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
type PjitSharding = GSPMDSharding | UnspecifiedValue | AUTO
type PjitShardingMinusUnspecified = GSPMDSharding | AUTO
type MeshSharding = NamedSharding | UnspecifiedValue | AUTO
type MeshShardingMinusUnspecified = NamedSharding | AUTO
logger: Incomplete

class PjitInfo(NamedTuple):
    fun_sourceinfo: str
    fun_signature: inspect.Signature | None
    user_specified_in_shardings: bool
    in_shardings_treedef: PyTreeDef
    in_shardings_leaves: tuple[Any, ...]
    out_shardings_treedef: PyTreeDef
    out_shardings_leaves: tuple[Any, ...]
    in_layouts_treedef: PyTreeDef
    in_layouts_leaves: tuple[Any, ...]
    out_layouts_treedef: PyTreeDef
    out_layouts_leaves: tuple[Any, ...]
    static_argnums: tuple[int, ...]
    static_argnames: tuple[str, ...]
    donate_argnums: tuple[int, ...]
    donate_argnames: tuple[str, ...]
    device: xc.Device | None
    backend: str | None
    keep_unused: bool
    inline: bool
    use_resource_env: bool
    compiler_options_kvs: tuple[tuple[str, Any], ...]
    def __hash__(self): ...
    def __eq__(self, other): ...

@api_boundary
def jit_trace(jit_func, *args, **kwargs) -> stages.Traced: ...
@api_boundary
def jit_lower(jit_func, *args, **kwargs): ...
@api_boundary
def jit_eval_shape(jit_func, *args, **kwargs): ...
def jit_evict_fn(self) -> None: ...
def make_jit(
    fun: Callable,
    *,
    in_shardings: Any,
    out_shardings: Any,
    static_argnums: int | Sequence[int] | None,
    static_argnames: str | Iterable[str] | None,
    donate_argnums: int | Sequence[int] | None,
    donate_argnames: str | Iterable[str] | None,
    keep_unused: bool,
    device: xc.Device | None,
    backend: str | None,
    inline: bool,
    compiler_options: dict[str, Any] | None,
    use_resource_env: bool,
) -> Any: ...

class PjitParams(NamedTuple):
    consts: list[ArrayLike]
    params: dict[str, Any]
    in_avals: tuple[core.AbstractValue, ...]
    in_tree: PyTreeDef
    out_tree: PyTreeDef
    arg_names: tuple[str, ...]

@dataclass(slots=True)
class InferParamsCacheEntry:
    pjit_params: PjitParams | None = ...

class JitWrapped(stages.Wrapped, metaclass=abc.ABCMeta):
    def eval_shape(self, *args, **kwargs) -> None: ...
    def trace(self, *args, **kwargs) -> stages.Traced: ...

def pjit(
    fun: Callable,
    in_shardings: Any = ...,
    out_shardings: Any = ...,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    keep_unused: bool = False,
    device: xc.Device | None = None,
    backend: str | None = None,
    inline: bool = False,
    compiler_options: dict[str, Any] | None = None,
) -> JitWrapped: ...
def hashable_pytree(pytree): ...
def flatten_axis_resources(what, tree, shardings, tupled_args): ...

class PytreeLeaf: ...

@dataclass(frozen=True)
class IgnoreKey:
    val: Any
    def __hash__(self): ...
    def __eq__(self, other): ...

def pjit_check_aval_sharding(
    shardings,
    flat_avals,
    names: Sequence[str],
    what_aval: str,
    allow_uneven_sharding: bool,
    allow_partial_manual: bool = False,
): ...
def check_aval_layout_compatibility(
    layouts,
    flat_avals,
    names: Sequence[str],
    what_aval: str,
): ...

jit_p: Incomplete

def finalize_arg_sharding(arg_s, committed): ...

@dataclass(frozen=True)
class MetaTy:
    aval: Any
    sharding: Any
    format: Any
    committed: bool
    is_np_array: bool
    replace = replace
    @property
    def shape(self): ...
    @property
    def ndim(self): ...

def create_meta_ty(aval, arg_sharding, arg_format, arg_committed, is_np_array): ...
def convert_to_metaty(arg): ...
def pjit_staging_rule(trace, source_info, *args, **params): ...
def pjit_forwarding_rule(eqn): ...
def const_args_shardings(const_args: Sequence[ArrayLike]) -> Sequence[PjitSharding]: ...
def const_args_layouts(
    const_args: Sequence[ArrayLike],
    avals: Sequence[core.AbstractValue],
    shardings: Sequence[PjitSharding],
) -> Sequence[Layout | AutoLayout | None]: ...
def dce_jaxpr_pjit_rule(
    used_outputs: list[bool],
    eqn: core.JaxprEqn,
) -> tuple[list[bool], core.JaxprEqn | None]: ...
def check_shardings_are_auto(s: Sharding) -> None: ...
def assert_shardings_equal(x_aval, user_sharding: NamedSharding): ...
def with_sharding_constraint(x, shardings): ...

sharding_constraint_p: Incomplete

def reshard(xs, out_shardings): ...

reshard_p: Incomplete

@dataclass(frozen=True, kw_only=True)
class MeshInfo:
    prev: AbstractMesh
    new: AbstractMesh
    axes: Any

def auto_axes(
    f=None,
    /,
    *,
    axes: str | tuple[str, ...] | None = None,
    out_sharding=None,
): ...
def explicit_axes(
    f=None,
    /,
    *,
    axes: str | tuple[str, ...] | None = None,
    in_sharding=None,
): ...
def with_layout_constraint(x, layouts): ...

layout_constraint_p: Incomplete

def get_unconstrained_dims(sharding: NamedSharding): ...
