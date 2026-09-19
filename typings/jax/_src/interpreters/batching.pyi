from collections.abc import Callable, Sequence
from typing import Any

import dataclasses

from _typeshed import Incomplete
from jax._src import (
    config as config,
    core as core,
    linear_util as lu,
    source_info_util as source_info_util,
)
from jax._src.ad_util import (
    SymbolicZero as SymbolicZero,
    Zero as Zero,
    add_jaxvals as add_jaxvals,
    add_jaxvals_p as add_jaxvals_p,
)
from jax._src.core import (
    Trace as Trace,
    Tracer as Tracer,
    TraceTag as TraceTag,
    typeof as typeof,
)
from jax._src.tree_util import (
    PyTreeDef as PyTreeDef,
    tree_flatten as tree_flatten,
    tree_unflatten as tree_unflatten,
)
from jax._src.typing import Array as Array
from jax._src.util import (
    as_hashable_function as as_hashable_function,
    canonicalize_axis as canonicalize_axis,
    memoize as memoize,
    moveaxis as moveaxis,
    safe_map as safe_map,
    safe_zip as safe_zip,
    split_list as split_list,
    tuple_insert as tuple_insert,
    unzip2 as unzip2,
    weakref_lru_cache as weakref_lru_cache,
)

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
type Vmappable = Any
type Elt = Any
type MapSpec = Any
type AxisSize = Any
type MeshAxis = Any
type GetIdx = Callable[[], Tracer]
type ToEltHandler = Callable[[Callable, GetIdx, Vmappable, MapSpec], Elt]
type FromEltHandler = Callable[[Callable, AxisSize, Elt, MapSpec], Vmappable]
type MakeIotaHandler = Callable[[AxisSize], Array]

def to_elt(trace: BatchTrace, get_idx: GetIdx, x: Vmappable, spec: MapSpec) -> Elt: ...

to_elt_handlers: dict[type, ToEltHandler]

def from_elt(
    trace: BatchTrace,
    axis_size: AxisSize,
    mesh_axis: MeshAxis,
    sum_match: bool,
    i: int,
    x: Elt,
    spec: MapSpec,
) -> tuple[Vmappable, MapSpec]: ...

from_elt_handlers: dict[type, FromEltHandler]

def make_iota(axis_size: AxisSize) -> Array: ...

make_iota_handlers: dict[type, MakeIotaHandler]

def register_vmappable(
    data_type: type,
    spec_type: type,
    axis_size_type: type,
    to_elt: Callable,
    from_elt: Callable,
    make_iota: Callable | None,
): ...

vmappables: dict[type, tuple[type, type]]
spec_types: set[type]

def unregister_vmappable(data_type: type) -> None: ...
def is_vmappable(x: Any) -> bool: ...
@lu.transformation_with_aux2
def flatten_fun_for_vmap(
    f: Callable,
    store: lu.Store,
    in_tree: PyTreeDef,
    *args_flat,
): ...

NotMapped: Incomplete
not_mapped: Incomplete

class BatchTracer(Tracer):
    val: Incomplete
    batch_dim: Incomplete
    source_info: Incomplete
    def __init__(
        self,
        trace: BatchTrace,
        val,
        batch_dim: NotMapped | int,
        source_info: source_info_util.SourceInfo | None = None,
    ) -> None: ...
    @property
    def aval(self): ...
    def full_lower(self): ...
    def get_referent(self): ...

@dataclasses.dataclass(frozen=True)
class AxisData:
    name: Any
    size: Any
    spmd_name: Any
    @property
    def explicit_mesh_axis(self): ...

def get_sharding_for_vmap(axis_data, orig_sharding, axis): ...

class BatchTrace(Trace):
    parent_trace: Incomplete
    axis_data: Incomplete
    tag: Incomplete
    requires_low: bool
    def __init__(self, parent_trace, tag, axis_data) -> None: ...
    def to_batch_info(self, val): ...
    def cur_qdd(self, x): ...
    def process_primitive(self, p, tracers, params): ...
    def process_call(self, call_primitive, f, tracers, params): ...
    def process_map(self, map_primitive, f: lu.WrappedFun, tracers, params): ...
    def process_custom_jvp_call(self, prim, fun, jvp, tracers, *, symbolic_zeros): ...
    def process_custom_vjp_call(
        self,
        prim,
        fun,
        fwd,
        bwd,
        tracers,
        *,
        out_trees,
        symbolic_zeros,
    ): ...

def batch(
    fun: lu.WrappedFun,
    axis_data,
    in_dims,
    out_dim_dests,
    sum_match: bool = False,
) -> lu.WrappedFun: ...
@lu.transformation_with_aux2
def batch_subtrace(f, store, tag, axis_data, in_dims, *in_vals): ...
def batch_jaxpr2(
    closed_jaxpr: core.ClosedJaxpr,
    axis_data,
    in_axes: tuple[int | NotMapped, ...],
) -> tuple[core.ClosedJaxpr, tuple[int | NotMapped]]: ...
def batch_jaxpr(closed_jaxpr, axis_data, in_batched, instantiate): ...
def batch_jaxpr_axes(closed_jaxpr, axis_data, in_axes, out_axes_dest): ...

class ZeroIfMapped: ...

zero_if_mapped: Incomplete

@lu.transformation_with_aux2
def batch_custom_jvp_subtrace(f, store, tag, axis_data, in_dims, *in_vals): ...
def batch_custom_vjp_bwd(
    bwd: lu.WrappedFun,
    tag: core.TraceTag,
    axis_data: AxisData,
    in_dims: Callable[[], Sequence[int | None]],
    out_dim_dests: Sequence[int | None],
) -> lu.WrappedFun: ...

fancy_primitive_batchers: dict[core.Primitive, Callable]

class AxisPrimitiveBatchersProxy:
    def __setitem__(self, prim, batcher) -> None: ...

axis_primitive_batchers: Incomplete

class PrimitiveBatchersProxy:
    def __setitem__(self, prim, batcher) -> None: ...
    def __delitem__(self, prim) -> None: ...

primitive_batchers: Incomplete

def defvectorized(prim) -> None: ...
def vectorized_batcher(prim, axis_data, batched_args, batch_dims, **params): ...
def defbroadcasting(prim) -> None: ...
def broadcast_batcher(prim, axis_data, args, dims, **params): ...
def defreducer(prim) -> None: ...
def reducer_batcher(prim, axis_data, batched_args, batch_dims, axes, **params): ...
def expand_dims_batcher(prim, args, dims, **params): ...
def broadcast(x, sz, axis, mesh_axis): ...
def matchaxis2(axis_data, src, dst, x, sum_match: bool = False): ...
def matchaxis(axis_name, sz, mesh_axis, src, dst, x, sum_match: bool = False): ...

class SpecMatchError(Exception):
    leaf_idx: Incomplete
    src: Incomplete
    dst: Incomplete
    def __init__(self, leaf_idx, src, dst) -> None: ...

def bdim_at_front(x, bdim, size, mesh_axis=None): ...
def add_batched(axis_data, batched_args, batch_dims): ...

class Sum: ...

sum_axis: Incomplete

class Infer: ...

infer: Incomplete
