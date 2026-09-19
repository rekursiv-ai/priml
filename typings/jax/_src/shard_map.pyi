from collections.abc import (
    Callable,
    Hashable,
    Set as AbstractSet,
)
from typing import Any, TypeVar, overload

from _typeshed import Incomplete
from jax._src import (
    ad_util as ad_util,
    api as api,
    api_util as api_util,
    config as config,
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
    sharding_impls as sharding_impls,
    source_info_util as source_info_util,
    traceback_util as traceback_util,
    util as util,
)
from jax._src.core import (
    Tracer as Tracer,
    order_wrt_mesh as order_wrt_mesh,
    pvary as pvary,
    shard_aval as shard_aval,
    typeof as typeof,
    unshard_aval as unshard_aval,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
    pxla as pxla,
)
from jax._src.lax import lax as lax
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import (
    hlo as hlo,
    sdy as sdy,
)
from jax._src.mesh import (
    AbstractMesh as AbstractMesh,
    AxisType as AxisType,
    BaseMesh as BaseMesh,
    Mesh as Mesh,
    get_abstract_mesh as get_abstract_mesh,
    get_concrete_mesh as get_concrete_mesh,
    use_abstract_mesh as use_abstract_mesh,
)
from jax._src.sharding_impls import (
    NamedSharding as NamedSharding,
    PartitionSpec as PartitionSpec,
)
from jax._src.state import discharge as discharge
from jax._src.state.types import AbstractRef as AbstractRef
from jax._src.tree_util import (
    KeyPath as KeyPath,
    PyTreeDef as PyTreeDef,
    broadcast_prefix as broadcast_prefix,
    generate_key_paths as generate_key_paths,
    keystr as keystr,
    prefix_errors as prefix_errors,
    tree_flatten as tree_flatten,
    tree_leaves as tree_leaves,
    tree_map as tree_map,
    tree_structure as tree_structure,
    tree_unflatten as tree_unflatten,
)
from jax._src.util import (
    HashableFunction as HashableFunction,
    HashablePartial as HashablePartial,
    as_hashable_function as as_hashable_function,
    merge_lists as merge_lists,
    partition_list as partition_list,
    split_list as split_list,
    subs_list2 as subs_list2,
    unzip2 as unzip2,
)

P = PartitionSpec
map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
type Specs = Any
AxisName = Hashable

class InferFromArgs:
    def __reduce__(self): ...

Infer: Incomplete
F = TypeVar("F", bound=Callable)
G = TypeVar("G", bound=Callable)

@overload
def shard_map(
    f: F,
    /,
    *,
    out_specs: Specs,
    in_specs: Specs | InferFromArgs | None = ...,
    mesh: Mesh | AbstractMesh | None = ...,
    axis_names: AbstractSet[AxisName] = ...,
    check_vma: bool = ...,
) -> F: ...
@overload
def shard_map(
    f: None = None,
    /,
    *,
    out_specs: Specs,
    in_specs: Specs | InferFromArgs | None = ...,
    mesh: Mesh | AbstractMesh | None = ...,
    axis_names: AbstractSet[AxisName] = ...,
    check_vma: bool = ...,
) -> Callable[[G], G]: ...
@overload
def smap(
    f: F,
    /,
    *,
    in_axes: int | InferFromArgs | tuple[Any, ...] | None = ...,
    out_axes: Any,
    axis_name: AxisName,
) -> F: ...
@overload
def smap(
    f: None = None,
    /,
    *,
    in_axes: int | InferFromArgs | tuple[Any, ...] | None = ...,
    out_axes: Any,
    axis_name: AxisName,
) -> Callable[[G], G]: ...

SpecErrorType: Incomplete

class NoFail: ...

no_fail: Incomplete
T = TypeVar("T")

class Tup:
    vals: Incomplete
    def __init__(self, vals) -> None: ...
    def __iter__(self): ...

type JaxType = Any
type MaybeTracer = JaxType | Tracer

class ShardMapPrimitive(core.Primitive):
    multiple_results: bool
    def bind(self, *args, **params): ...
    def bind_with_trace(self, trace, fun_and_args, params): ...
    def get_bind_params(self, params): ...

shard_map_p: Incomplete

def get_mesh_from_args(args_flat, mesh): ...

class _SpecError(Exception): ...
class _RepError(Exception): ...

class ShardMapTrace(core.Trace):
    mesh: Mesh
    manual_axes: frozenset[AxisName]
    check: bool
    amesh: Incomplete
    def __init__(self, mesh, manual_axes, check) -> None: ...
    def to_val_vma_pair(self, val): ...
    def process_primitive(self, prim, tracers, params): ...
    def process_shard_map(
        self,
        prim,
        fun,
        args,
        mesh,
        in_specs,
        out_specs_thunk,
        check_vma,
        manual_axes,
    ): ...
    def process_call(self, call_primitive, fun, tracers, params) -> None: ...
    def process_map(self, map_primitive, fun, tracers, params) -> None: ...
    def process_custom_jvp_call(self, prim, fun, jvp, tracers, *, symbolic_zeros): ...
    def process_custom_vjp_call(
        self,
        prim,
        fun,
        fwd,
        bwd,
        tracers,
        out_trees,
        symbolic_zeros,
    ): ...

class ShardMapTracer(core.Tracer):
    vma: frozenset[AxisName]
    val: JaxType
    def __init__(self, trace, vma, val) -> None: ...
    @property
    def aval(self): ...
    def to_concrete_value(self): ...

eager_rules: dict[core.Primitive, Callable]

def raise_notimplemented(*args, **kwargs) -> None: ...
