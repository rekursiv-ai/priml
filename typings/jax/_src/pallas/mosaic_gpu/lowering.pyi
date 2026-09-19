from collections.abc import (
    Callable as Callable,
    Hashable,
    Iterator,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from typing import Any, Protocol, Self, TypeVar

import collections
import contextlib
import dataclasses
import functools

from _typeshed import Incomplete
from jax import (
    api_util as api_util,
    lax as lax,
)
from jax._src import (
    checkify as checkify,
    config as config,
    core as jax_core,
    debugging as debugging,
    dtypes as dtypes,
    literals as literals,
    mesh as mesh_lib,
    pjit as pjit,
    source_info_util as source_info_util,
    tree_util as tree_util,
    util as util,
)
from jax._src.interpreters import mlir as mlir
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir as ir
from jax._src.pallas import (
    core as pallas_core,
    primitives as primitives,
    utils as pallas_utils,
)
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.state import (
    discharge as discharge,
    indexing as indexing,
    types as state_types,
)
from jax._src.state.types import (
    ReshapeTransform as ReshapeTransform,
    TransposeTransform as TransposeTransform,
)
from jax._src.util import foreach as foreach
from jax.experimental.mosaic.gpu import (
    profiler as mgpu_profiler,
    tcgen05 as tcgen05,
)

import jax
import jax.experimental.mosaic.gpu as mgpu
import jax.numpy as jnp

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
partial = functools.partial
SMEM: Incomplete
WARPGROUP_SIZE: int
RefOrTmemType = TypeVar("RefOrTmemType", ir.Value, tcgen05.TMEMRef)
type CollectiveAxesType = Sequence[Hashable]

@dataclasses.dataclass(frozen=True, kw_only=True)
class ResourceEstimatorContext:
    reduction_scratch_bytes: int
    axis_names: _AxisNames
    lowering_semantics: mgpu.LoweringSemantics
    @property
    def arrival_multiplier(self) -> int: ...

type AnyBarrier = mgpu.Barrier | mgpu.ClusterBarrier

@dataclasses.dataclass(kw_only=True, frozen=True)
class Resources:
    smem_scratch_bytes: int = ...
    tmem_scratch_cols: int = ...
    tmem_collective_scratch_cols: int = ...
    barrier_counts: collections.Counter[AnyBarrier] = ...
    scoped_gmem_semaphores: dict[CollectiveAxesType, int] = ...
    def __post_init__(self) -> None: ...
    @property
    def barriers(self) -> Sequence[AnyBarrier]: ...
    def __add__(self, other: Resources) -> Resources: ...
    def or_(self, other: Resources, axis_names: _AxisNames) -> Resources: ...

class ResourceEstimator(Protocol):
    def __call__(
        self,
        ctx: ResourceEstimatorContext,
        *args: Any,
        **params: Any,
    ) -> Resources: ...

@dataclasses.dataclass(frozen=True)
class _AxisNames:
    grid: Sequence[Hashable]
    cluster: Sequence[Hashable] = ...
    wg: Hashable | None = ...
    def __iter__(self) -> Iterator[Hashable]: ...
    def reverse(self) -> _AxisNames: ...

type AnyBarrierRef = (
    mgpu.BarrierRef | mgpu.DialectBarrierRef | mgpu.CollectiveBarrierRef
)

@dataclasses.dataclass
class ModuleContext:
    name: str
    axis_names: _AxisNames
    program_ids: Sequence[ir.Value] | None
    approx_math: bool
    single_wg_lane_predicate: ir.Value | None
    single_warp_lane_predicate: ir.Value | None
    smem_requested_bytes: int
    smem_used_bytes: int
    tmem_requested_cols: int
    tmem_used_cols: int
    tmem_base: ir.Value | None
    scoped_gmem_used_semaphores: dict[CollectiveAxesType, int]
    scoped_gmem_semaphore_base_ptr: dict[CollectiveAxesType, ir.Value]
    runtime_barriers: MutableMapping[AnyBarrier, MutableSequence[AnyBarrierRef]]
    name_stack: source_info_util.NameStack
    traceback_caches: mlir.TracebackCaches
    squashed_dims: tuple[int, ...]
    lowering_semantics: mgpu.LoweringSemantics
    primitive_semantics: gpu_core.PrimitiveSemantics
    mesh_info: pallas_utils.MeshInfo | None
    auto_barriers: bool
    reduction_scratch_bytes: int
    warp_axis_name: str | None = ...
    outer_traceback: xc.Traceback | None = ...
    @property
    def single_lane_predicate(self) -> ir.Value: ...
    @contextlib.contextmanager
    def reserve_barrier(
        self,
        barrier: mgpu.Barrier | mgpu.ClusterBarrier,
    ) -> Iterator[
        mgpu.BarrierRef | mgpu.DialectBarrierRef | mgpu.CollectiveBarrierRef
    ]: ...
    @contextlib.contextmanager
    def reserve_semaphores(
        self,
        shape: tuple[int, ...],
        collective_axes: CollectiveAxesType,
    ) -> Iterator[ir.Value]: ...
    @contextlib.contextmanager
    def alloc_tmem(
        self,
        struct: jax.ShapeDtypeStruct,
        *,
        layout: tcgen05.TMEMLayout,
    ) -> Iterator[tcgen05.TMEMRef | ir.Value]: ...
    @contextlib.contextmanager
    def scratch_view(self, struct: jax.ShapeDtypeStruct) -> Iterator[ir.Value]: ...

class ShapedAbstractValue(Protocol):
    shape: tuple[jax_core.DimSize, ...]
    dtype: jnp.dtype
    weak_type: bool
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    def update(self, **kwargs: Any) -> Self: ...

@dataclasses.dataclass(frozen=True)
class LoweringRuleContext:
    module_ctx: ModuleContext
    launch_ctx: mgpu.LaunchContext
    prim: jax_core.Primitive
    avals_in: Sequence[ShapedAbstractValue]
    avals_out: Sequence[ShapedAbstractValue]
    out_layout_hint: mgpu.FragmentedLayout | None
    def replace(self, **changes: Any) -> LoweringRuleContext: ...
    @property
    def estimator_ctx(self) -> ResourceEstimatorContext: ...

@dataclasses.dataclass(frozen=True)
class LoweringResult:
    module: ir.Module
    grid: tuple[int, ...]
    block: tuple[int, ...]
    new_out_shapes: tuple[jax.ShapeDtypeStruct, ...]
    profiler_spec: mgpu_profiler.ProfilerSpec | None
    gmem_scratch_shapes: tuple[jax.ShapeDtypeStruct, ...]

class LoweringError(Exception): ...

def lower_pipelined_jaxpr_to_module(
    grid_mapping: pallas_core.GridMapping,
    gpu_mesh: gpu_core.Mesh | None,
    jax_mesh: mesh_lib.Mesh | None,
    jaxpr: jax_core.Jaxpr,
    params: gpu_core.CompilerParams,
    cost_estimate: pallas_core.CostEstimate | None,
    outer_traceback: xc.Traceback | None = None,
) -> LoweringResult: ...
def lower_jaxpr_to_module(
    jax_mesh: mesh_lib.Mesh | None,
    axis_names: _AxisNames,
    grid: tuple[int, ...],
    block: tuple[int, int, int],
    cluster: tuple[int, ...],
    in_shapes: Sequence[jax_core.ShapedArray],
    out_shapes: Sequence[jax_core.ShapedArray],
    jaxpr: jax_core.Jaxpr,
    params: gpu_core.CompilerParams,
    consts=(),
    outer_traceback: xc.Traceback | None = None,
) -> LoweringResult: ...

mosaic_lowering_rules: Incomplete

def register_lowering_rule(
    primitive: jax_core.Primitive,
    lowering_semantics: mgpu.LoweringSemantics,
    primitive_semantics: gpu_core.PrimitiveSemantics = ...,
): ...
def lower_jaxpr_to_mosaic_gpu(
    module_ctx: ModuleContext,
    launch_ctx: mgpu.LaunchContext,
    jaxpr: jax_core.Jaxpr,
    args: Sequence[ir.Value],
    consts=(),
) -> Sequence[ir.Value]: ...

T = TypeVar("T", bound=state_types.Transform)
CmpIPred: Incomplete
CmpFPred: Incomplete

def block_id_to_grid_id(
    ctx: LoweringRuleContext,
    block_ids: Sequence[ir.Value],
    axis_name: Hashable,
): ...
def merge_indexers(indexers: Sequence[indexing.NDIndexer]) -> indexing.NDIndexer: ...
