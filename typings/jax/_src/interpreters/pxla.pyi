from collections.abc import Callable, Iterable, Sequence
from functools import cached_property as cached_property
from typing import Any, NamedTuple

import dataclasses
import functools

from _typeshed import Incomplete
from jax._src import (
    api as api,
    array as array,
    compiler as compiler,
    config as config,
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
    effects as effects,
    jaxpr_util as jaxpr_util,
    linear_util as lu,
    literals as literals,
    op_shardings as op_shardings,
    pjit as pjit,
    profiler as profiler,
    sharding_impls as sharding_impls,
    sharding_specs as sharding_specs,
    stages as stages,
    tree_util as tree_util,
    typing as typing,
    util as util,
)
from jax._src.abstract_arrays import array_types as array_types
from jax._src.core import ShapedArray as ShapedArray
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)
from jax._src.layout import (
    AutoLayout as AutoLayout,
    Format as Format,
    Layout as Layout,
)
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import hlo as hlo
from jax._src.mesh import (
    AbstractMesh as AbstractMesh,
    Mesh as Mesh,
    get_abstract_mesh as get_abstract_mesh,
    get_concrete_mesh as get_concrete_mesh,
)
from jax._src.partition_spec import PartitionSpec as PartitionSpec
from jax._src.sharding import (
    IndivisibleError as IndivisibleError,
    Sharding as JSharding,
)
from jax._src.sharding_impls import (
    AUTO as AUTO,
    ArrayMapping as ArrayMapping,
    GSPMDSharding as GSPMDSharding,
    NamedSharding as NamedSharding,
    SingleDeviceSharding as SingleDeviceSharding,
    UnspecifiedValue as UnspecifiedValue,
    array_mapping_to_axis_resources as array_mapping_to_axis_resources,
)
from jax._src.state.types import (
    AbstractRef as AbstractRef,
    RefEffect as RefEffect,
)
from jax._src.typing import ArrayLike as ArrayLike
from jax._src.util import (
    HashableFunction as HashableFunction,
    distributed_debug_log as distributed_debug_log,
    partition_list as partition_list,
    safe_map as safe_map,
    safe_zip as safe_zip,
    tuple_insert as tuple_insert,
    tuple_update as tuple_update,
    unzip2 as unzip2,
    weakref_lru_cache as weakref_lru_cache,
    wrap_name as wrap_name,
)

class WeakRefList(list): ...

unsafe_map: Incomplete
map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
logger: Incomplete
type Index = int | slice | tuple[int | slice, ...]
PyTreeDef: Incomplete
NoSharding: Incomplete
Chunked: Incomplete
Unstacked: Incomplete
ShardedAxis: Incomplete
Replicated: Incomplete
type AvalDimSharding = Unstacked | Chunked | NoSharding
MeshAxisName: Incomplete
type MeshDimAssignment = ShardedAxis | Replicated
ShardingSpec: Incomplete

def identity(x): ...
@profiler.annotate_function
def shard_args(
    shardings: Sequence[JSharding],
    layouts: Sequence[Any | None],
    copy_semantics: Sequence[xc.ArrayCopySemantics],
    args: Sequence[Any],
    canonicalize: bool = True,
) -> Sequence[xc.ArrayImpl]: ...

shard_arg_handlers: dict[
    Any,
    Callable[
        [Sequence[Any], Sequence[Any], Sequence[Any], Sequence[xc.ArrayCopySemantics]],
        Sequence[Any],
    ],
]

def is_default_layout(curr_layout, sharding, aval): ...
def batched_device_put(
    aval: core.ShapedArray,
    sharding: JSharding,
    xs: Sequence[Any],
    devices: Sequence[xc.Device],
    committed: bool = True,
    enable_x64: bool | None = None,
): ...
def local_aval_to_result_handler(
    aval: core.AbstractValue,
    sharding: JSharding,
    indices: tuple[Index, ...] | None,
) -> Callable[[list[xc.ArrayImpl]], Any]: ...

PxlaResultHandler: Incomplete
local_result_handlers: dict[type[core.AbstractValue], PxlaResultHandler]

def global_aval_to_result_handler(
    aval: core.AbstractValue,
    out_sharding,
    committed: bool,
) -> Callable[[Sequence[xc.ArrayImpl]], Any]: ...

global_result_handlers: dict[type[core.AbstractValue], PxlaResultHandler]

def xla_pmap_impl_lazy(
    fun: lu.WrappedFun,
    *args,
    backend: str | None,
    axis_name: core.AxisName,
    axis_size: int,
    global_axis_size: int,
    devices: Sequence[Any] | None,
    name: str,
    in_axes: Sequence[int | None],
    out_axes_thunk: Callable[[], Sequence[int | None]],
    donated_invars: Sequence[bool],
    is_explicit_global_axis_size: bool,
) -> tuple[Callable, list[ArrayLike]]: ...
def xla_pmap_impl(fun: lu.WrappedFun, *args, **params): ...

class EmapInfo(NamedTuple):
    backend: str | None
    devices: Sequence[Any] | None

class FakePrimitive(NamedTuple):
    multiple_results: Incomplete
    bind: Incomplete

class MapTrace(core.Trace):
    emap_info: Incomplete
    axis_name: Incomplete
    def __init__(self, axis_name, emap_info) -> None: ...
    def to_map_tracer(self, val): ...
    def process_primitive(self, primitive, tracers, params): ...
    def process_call(self, call_primitive, fun, tracers, params) -> None: ...
    def process_map(self, map_primitive, fun, tracers, params): ...
    def process_custom_jvp_call(self, prim, fun, jvp, tracers, *, symbolic_zeros): ...
    def process_custom_vjp_call(
        self,
        primitive,
        fun,
        fwd,
        bwd,
        tracers,
        out_trees,
        symbolic_zeros,
    ): ...
    def process_axis_index(self, axis_name): ...

class MapTracer(core.Tracer):
    val: Incomplete
    shard_axes: Incomplete
    def __init__(
        self,
        trace: MapTrace,
        val,
        shard_axes: dict[core.AxisName, int],
    ) -> None: ...
    @property
    def aval(self): ...
    def full_lower(self): ...

@lu.cache
def parallel_callable(
    fun: lu.WrappedFun,
    backend_name: str | None,
    axis_name: core.AxisName,
    axis_size: int,
    global_axis_size: int,
    devices: Sequence[Any] | None,
    name: str,
    in_axes: Sequence[int | None],
    out_axes_thunk: Callable[[], Sequence[int | None]],
    donated_invars: Sequence[bool],
    is_explicit_global_axis_size: bool,
    *avals,
): ...

@dataclasses.dataclass(frozen=True)
class ParallelCallableInfo:
    name: str
    backend: xc.Client
    axis_name: core.AxisName
    axis_size: int
    global_axis_size: int
    devices: Sequence[xc.Device] | None
    in_axes: Iterable[int | None]
    out_axes_thunk: Callable[[], Sequence[int | None]]
    avals: Sequence[core.AbstractValue]
    @cached_property
    def local_devices(self): ...
    @cached_property
    def out_axes(self): ...

class ShardInfo(NamedTuple):
    sharded_avals: Sequence[core.AbstractValue]
    out_sharded_avals: Sequence[core.ShapedArray]
    global_sharded_avals: Sequence[core.AbstractValue]
    num_local_shards: int
    num_global_shards: int

class ReplicaInfo(NamedTuple):
    jaxpr_replicas: int
    num_local_replicas: int
    num_global_replicas: int

def register_initial_style_primitive(prim: core.Primitive): ...
def find_replicas(
    jaxpr: core.Jaxpr,
    axis_size: int,
    global_axis_size: int,
) -> ReplicaInfo: ...
def stage_parallel_callable(
    pci: ParallelCallableInfo,
    fun: lu.WrappedFun,
) -> tuple[core.Jaxpr, list[Any], ReplicaInfo, ShardInfo]: ...
def get_pmap_jaxpr(
    fun: lu.WrappedFun,
    backend_name: str | None,
    axis_name: core.AxisName,
    axis_size: int,
    global_axis_size: int,
    devices: Sequence[xc.Device] | None,
    name: str,
    in_axes: Iterable[int | None],
    out_axes_thunk: Callable[[], Sequence[int | None]],
    avals: Sequence[core.AbstractValue],
): ...
@profiler.annotate_function
def lower_parallel_callable(
    fun: lu.WrappedFun,
    axis_name: core.AxisName,
    axis_size: int,
    global_axis_size: int,
    devices: Sequence[xc.Device] | None,
    name: str,
    in_axes: Iterable[int | None],
    donated_invars: Sequence[bool],
    is_explicit_global_axis_size: bool,
    avals: Sequence[core.AbstractValue],
    *,
    lowering_platforms: tuple[str, ...] | None,
    lowering_parameters: mlir.LoweringParameters,
    closed_jaxpr: core.ClosedJaxpr,
    backend: xc.Client,
    replicas: ReplicaInfo,
    shards: ShardInfo,
    pci: ParallelCallableInfo,
) -> PmapComputation: ...

type AvalMapHandlerPair = tuple[Any, Callable]

class PmapComputation(stages.Lowering):
    const_args: Incomplete
    compile_args: Incomplete
    def __init__(
        self,
        hlo: ir.Module,
        const_args: list[ArrayLike],
        **compile_args,
    ) -> None: ...
    def stablehlo(self) -> ir.Module: ...
    @profiler.annotate_function
    def compile(
        self,
        compiler_options=None,
        *,
        device_assignment=None,
    ) -> PmapExecutable: ...

@dataclasses.dataclass
class UnloadedPmapExecutable:
    compiled: Any
    backend: xc.Client
    local_input_avals: Sequence[core.AbstractValue]
    input_shardings: Sequence[JSharding]
    local_output_avals: Sequence[ShapedArray]
    output_shardings: Sequence[JSharding]
    unordered_effects: list[core.Effect]
    ordered_effects: list[core.Effect]
    keepalive: Sequence[Any]
    host_callbacks: Sequence[Any]
    jaxpr_debug_info: core.DebugInfo
    def build_execute_fun(self): ...
    def load(self) -> PmapExecutable: ...
    @staticmethod
    def from_hlo(
        hlo: ir.Module,
        pci: ParallelCallableInfo,
        replicas: ReplicaInfo,
        shards: ShardInfo,
        tuple_args: bool,
        unordered_effects: list[core.Effect],
        ordered_effects: list[core.Effect],
        host_callbacks: list[Any],
        keepalive: Any,
        jaxpr_debug_info: core.DebugInfo,
        platforms: Sequence[str],
        shape_poly_state: mlir.ShapePolyLoweringState | None = None,
        compiler_options=None,
    ): ...

class PmapExecutable(stages.Executable):
    xla_executable: Incomplete
    build_unsafe_call: Incomplete
    fingerprint: Incomplete
    in_avals: Incomplete
    def __init__(
        self,
        xla_executable,
        build_unsafe_call,
        fingerprint,
        in_avals,
        unloaded_executable: UnloadedPmapExecutable,
    ) -> None: ...
    @property
    def unsafe_call(self) -> Callable[..., Any]: ...
    def xla_extension_executable(self): ...
    @profiler.annotate_function
    def call(self, *args): ...

class InputsHandler:
    handler: Incomplete
    in_shardings: Incomplete
    in_layouts: Incomplete
    local_devices: Incomplete
    input_indices: Incomplete
    def __init__(
        self,
        in_shardings,
        in_layouts,
        local_devices=None,
        input_indices=None,
    ) -> None: ...
    def __call__(self, input_buffers): ...

class ResultsHandler:
    handlers: Incomplete
    out_shardings: Incomplete
    out_avals: Incomplete
    def __init__(self, handlers, out_shardings, out_avals) -> None: ...
    def __call__(self, out_bufs): ...

def local_avals_to_results_handler(
    unmapped_local_out_avals: Sequence[ShapedArray],
    local_shardings: Sequence[JSharding],
) -> ResultsHandler: ...
def global_avals_to_results_handler(
    global_out_avals: Sequence[ShapedArray],
    shardings: Sequence[JSharding],
    committed: bool,
) -> ResultsHandler: ...

class ExecuteReplicated:
    xla_executable: Incomplete
    name: Incomplete
    backend: Incomplete
    in_handler: Incomplete
    out_handler: Incomplete
    has_unordered_effects: Incomplete
    ordered_effects: Incomplete
    keepalive: Incomplete
    has_host_callbacks: Incomplete
    kept_var_idx: Incomplete
    mut: Incomplete
    pgle_profiler: Incomplete
    def __init__(
        self,
        xla_executable,
        name,
        backend,
        in_handler: InputsHandler,
        out_handler: ResultsHandler,
        unordered_effects: list[core.Effect],
        ordered_effects: list[core.Effect],
        keepalive: Any,
        has_host_callbacks: bool,
        kept_var_idx: set[int],
        mut: MutationData | None,
        pgle_profiler: profiler.PGLEProfiler | None = None,
    ) -> None: ...
    @profiler.annotate_function
    def __call__(self, *args): ...

xla_pmap_p: Incomplete
xla_pmap: Incomplete

def xla_call_jvp_update_params(params, nz_tangents): ...
def axis_groups(axis_env: sharding_impls.AxisEnv, name) -> tuple[tuple[int, ...]]: ...
def tile_aval_nd(axis_sizes, in_axes: ArrayMapping, aval): ...
def untile_aval_nd(axis_sizes, out_axes: ArrayMapping, aval): ...
def mesh_local_to_global(mesh, axes: ArrayMapping, aval): ...
def mesh_global_to_local(mesh, axes: ArrayMapping, aval): ...

full_to_shard_p: Incomplete

def manual_proto(
    aval: core.ShapedArray,
    manual_axes_set: frozenset[sharding_impls.MeshAxisName],
    mesh: Mesh,
): ...

shard_to_full_p: Incomplete

def check_if_any_auto(
    shardings: Iterable[JSharding | AUTO | UnspecifiedValue],
) -> bool: ...

ShardingInfo: Incomplete

def get_default_device() -> xc.Device: ...

type MaybeSharding = JSharding | UnspecifiedValue

def prune_unused_inputs(jaxpr: core.Jaxpr) -> tuple[core.Jaxpr, set[int], set[int]]: ...

class MutationData(NamedTuple):
    in_mut: list[core.Ref]
    out_mut: list[int | None]

class SemanticallyEqualShardings:
    shardings: Incomplete
    def __init__(
        self,
        shardings: tuple[GSPMDSharding | UnspecifiedValue, ...],
        avals: Sequence[core.AbstractValue],
    ) -> None: ...
    def __hash__(self): ...
    def __eq__(self, other): ...

@weakref_lru_cache
def jaxpr_transfer_mem_kinds(jaxpr: core.Jaxpr): ...
def are_all_shardings_default_mem_kind(shardings): ...
@weakref_lru_cache
def get_out_layouts_via_propagation(
    closed_jaxpr: core.ClosedJaxpr,
) -> tuple[Layout | None]: ...

type MaybeLayout = Sequence[Layout | AutoLayout | None]

class AllArgsInfo(NamedTuple):
    in_avals: Sequence[core.ShapedArray]
    debug_info: core.DebugInfo

def to_gspmd_sharding(s: JSharding, ndim: int) -> GSPMDSharding: ...
def hoist_constants_as_args(
    closed_jaxpr: core.ClosedJaxpr,
    global_in_avals,
    in_shardings,
    in_layouts,
    donated_invars,
    kept_var_idx: set[int],
    inout_aliases,
    mut,
    all_args_info: AllArgsInfo,
): ...
@profiler.annotate_function
def lower_sharding_computation(
    closed_jaxpr: core.ClosedJaxpr,
    api_name: str,
    fun_name: str,
    in_shardings: Sequence[MaybeSharding],
    out_shardings: Sequence[MaybeSharding],
    in_layouts: MaybeLayout,
    out_layouts: MaybeLayout,
    donated_invars: Sequence[bool],
    *,
    keep_unused: bool,
    context_mesh: Mesh,
    compiler_options_kvs: tuple[tuple[str, Any], ...],
    lowering_platforms: tuple[str, ...] | None,
    lowering_parameters: mlir.LoweringParameters,
    pgle_profiler: profiler.PGLEProfiler | None,
) -> MeshComputation: ...

class MeshComputation(stages.Lowering):
    const_args: Incomplete
    compile_args: Incomplete
    def __init__(
        self,
        name: str,
        hlo: ir.Module,
        const_args: list[ArrayLike],
        donated_invars: Sequence[bool],
        platforms: Sequence[str],
        compiler_options_kvs: tuple[tuple[str, Any], ...],
        device_assignment: xc.DeviceList | tuple[xc.Device, ...] | None,
        **compile_args,
    ) -> None: ...
    def stablehlo(self) -> ir.Module: ...
    def compile(
        self,
        compiler_options=None,
        *,
        device_assignment=None,
    ) -> MeshExecutable: ...
    def cost_analysis(self) -> dict[str, float]: ...

def get_op_sharding_from_executable(
    executable,
) -> tuple[Sequence[xc.OpSharding], Sequence[xc.OpSharding]]: ...
def get_pspec_from_executable(
    executable,
    mesh: Mesh,
) -> tuple[tuple[PartitionSpec, ...], tuple[PartitionSpec, ...]]: ...
def get_out_shardings_from_executable(
    xla_executable,
    device_list: xc.DeviceList,
    num_out_avals: int,
    num_ordered_effects: int,
) -> Sequence[sharding_impls.GSPMDSharding] | None: ...
def maybe_recover_user_shardings(
    old_shardings,
    new_shardings,
    old_avals,
    new_avals,
    intermediate_shardings=None,
    context_mesh: Mesh | None = None,
): ...
def is_user_xla_layout_equal(ul: Layout | AutoLayout, xl: Layout) -> bool: ...
def get_logical_mesh_ids(mesh_shape): ...
def create_compile_options(
    computation,
    mesh,
    spmd_lowering,
    tuple_args,
    auto_spmd_lowering,
    allow_prop_to_inputs,
    allow_prop_to_outputs,
    backend,
    np_dev,
    pmap_nreps,
    compiler_options,
): ...
def finalize_shardings(shardings, device_assignment): ...
def get_prop_to_input_output(in_shardings, out_shardings, num_ordered_effects): ...
def maybe_concretize_mesh(sharding, da: xc.DeviceList): ...

@dataclasses.dataclass
class UnloadedMeshExecutable:
    xla_executable: Any
    device_list: xc.DeviceList
    backend: xc.Client
    input_avals: Sequence[ShapedArray]
    input_shardings: Sequence[JSharding]
    output_avals: Sequence[ShapedArray]
    output_shardings: Sequence[JSharding]
    committed: bool
    name: str
    unordered_effects: list[core.Effect]
    ordered_effects: list[core.Effect]
    keepalive: Sequence[Any]
    host_callbacks: Sequence[Any]
    kept_var_idx: set[int]
    mut: MutationData | None
    auto_spmd_lowering: bool
    xla_in_layouts: Sequence[Layout | None]
    dispatch_in_layouts: Sequence[Layout | None]
    xla_out_layouts: Sequence[Layout | None]
    all_args_info: AllArgsInfo | None
    pgle_profiler: profiler.PGLEProfiler | None
    def build_unsafe_call(self): ...
    def load(self) -> MeshExecutable: ...
    @staticmethod
    def from_hlo(
        name: str,
        hlo: ir.Module,
        global_in_avals: Sequence[ShapedArray],
        global_out_avals: Sequence[ShapedArray],
        in_shardings: Sequence[JSharding | AUTO],
        out_shardings: Sequence[JSharding | AUTO | UnspecifiedValue],
        spmd_lowering: bool,
        tuple_args: bool,
        auto_spmd_lowering: bool,
        unordered_effects: list[core.Effect],
        ordered_effects: list[core.Effect],
        host_callbacks: list[Any],
        keepalive: Any,
        kept_var_idx: set[int],
        backend: xc.Client,
        device_list: xc.DeviceList | None,
        committed: bool,
        in_layouts: MaybeLayout,
        out_layouts: MaybeLayout,
        compiler_options_kvs: tuple[tuple[str, Any], ...],
        num_devices: int,
        pmap_nreps: int = 1,
        mut: MutationData | None = None,
        shape_poly_state: mlir.ShapePolyLoweringState | None = None,
        all_args_info: AllArgsInfo | None = None,
        pgle_profiler: profiler.PGLEProfiler | None = None,
        intermediate_shardings: Sequence[JSharding] | None = None,
        context_mesh: Mesh | None = None,
    ) -> MeshExecutable: ...

class MeshExecutableFastpathData(NamedTuple):
    xla_executable: xc.LoadedExecutable
    out_pytree_def: Any
    in_shardings: Sequence[JSharding]
    out_shardings: Sequence[JSharding]
    out_avals: Sequence[ShapedArray]
    out_committed: Sequence[bool]
    kept_var_bitvec: Iterable[bool]
    in_device_local_layouts: Sequence[Layout | None]
    const_args: Sequence[ArrayLike]

def clear_in_memory_compilation_cache() -> None: ...

@dataclasses.dataclass(frozen=True, kw_only=True)
class JitGlobalCppCacheKeys:
    donate_argnums: tuple[int, ...] | None = ...
    donate_argnames: tuple[str, ...] | None = ...
    device: xc.Device | None = ...
    backend: str | None = ...
    in_shardings_treedef: PyTreeDef | None = ...
    in_shardings_leaves: tuple[Any, ...] | None = ...
    out_shardings_treedef: PyTreeDef | None = ...
    out_shardings_leaves: tuple[Any, ...] | None = ...
    in_layouts_treedef: PyTreeDef | None = ...
    in_layouts_leaves: tuple[Any, ...] | None = ...
    out_layouts_treedef: PyTreeDef | None = ...
    out_layouts_leaves: tuple[Any, ...] | None = ...
    compiler_options_kvs: tuple[tuple[str, Any], ...] | None = ...
    @functools.cached_property
    def contains_explicit_attributes(self): ...

def reflatten_outputs_for_dispatch(out_tree, out_flat): ...

class MeshExecutable(stages.Executable):
    xla_executable: Incomplete
    build_unsafe_call: Incomplete
    in_avals: Incomplete
    out_avals: Incomplete
    def __init__(
        self,
        xla_executable,
        build_unsafe_call,
        in_avals,
        out_avals,
        in_shardings,
        out_shardings,
        auto_spmd_lowering,
        kept_var_idx,
        xla_in_layouts,
        dispatch_in_layouts,
        xla_out_layouts,
        mut,
        all_args_info: AllArgsInfo | None = None,
        unloaded_executable=None,
    ) -> None: ...
    @property
    def unsafe_call(self) -> Callable[..., Any]: ...
    def xla_extension_executable(self): ...
    def call(self, *args): ...
    def create_cpp_call(self, params: stages.CompiledCallParams): ...

def cc_shard_arg(x, sharding, layout): ...
def check_arg_avals_for_call(
    ref_avals,
    arg_avals,
    jaxpr_debug_info: core.DebugInfo,
): ...
def check_device_backend_on_shardings(shardings) -> bool: ...
def check_array_xla_sharding_layout_match(
    args,
    in_shardings: Sequence[JSharding],
    in_layouts: Sequence[Layout],
    arg_names: Sequence[str],
) -> None: ...
def batch_spec(spec, dim, val): ...
