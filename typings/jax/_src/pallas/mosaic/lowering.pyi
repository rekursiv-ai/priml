from collections.abc import Callable, Collection, Generator, Hashable, Sequence
from typing import Any, Protocol, Self, TypeVar

import contextlib
import dataclasses
import functools

from _typeshed import Incomplete
from jax import (
    api_util as api_util,
    lax as lax,
    tree_util as tree_util,
)
from jax._src import (
    ad_util as ad_util,
    checkify as checkify,
    config as config,
    core as jax_core,
    custom_derivatives as custom_derivatives,
    debugging as debugging,
    dtypes as dtypes,
    literals as literals,
    mesh as mesh_lib,
    pjit as pjit,
    prng as prng,
    source_info_util as source_info_util,
    state as state,
    traceback_util as traceback_util,
    xla_bridge as xla_bridge,
)
from jax._src.cloud_tpu_init import is_cloud_tpu_older_than as is_cloud_tpu_older_than
from jax._src.export import shape_poly as shape_poly
from jax._src.export._export import export as export
from jax._src.interpreters import mlir as mlir
from jax._src.lax import control_flow as control_flow
from jax._src.lax.control_flow import BranchesPlatforms as BranchesPlatforms
from jax._src.lib import xla_client as xla_client
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import (
    arith as arith,
    cf as cf,
    func as func,
    memref as memref,
    scf as scf,
    vector as vector,
)
from jax._src.pallas import (
    core as pallas_core,
    primitives as primitives,
    utils as pallas_utils,
)
from jax._src.pallas.mosaic import (
    core as tpu_core,
    error_handling as error_handling,
)
from jax._src.state import indexing as indexing
from jax._src.state.types import (
    BitcastTransform as BitcastTransform,
    ReshapeTransform as ReshapeTransform,
)
from jax._src.typing import (
    Array as Array,
    DTypeLike as DTypeLike,
)
from jax._src.util import (
    foreach as foreach,
    safe_map as safe_map,
    safe_zip as safe_zip,
    split_list as split_list,
)
from jax.experimental.mosaic.dialects import tpu as tpu

import jax.numpy as jnp

NDIndexer: Incomplete
TPUMemorySpace: Incomplete
AnyMemorySpace: Incomplete
VMEM: Incomplete
SMEM: Incomplete
ANY: Incomplete
BOOL_MEMREF_TYPE: Incomplete
MLIR_DYNAMIC: int
DIM_UPPER_BOUND: Incomplete
DIM_LOWER_BOUND: int
partial = functools.partial
map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
PHYSICAL_EXTENDED_DTYPES: Incomplete

def should_physicalize_dtype(dtype: DTypeLike) -> bool: ...

class LoweringDynamicShapeEnv:
    dim_expr_to_placeholder: dict[shape_poly._DimExpr, int]
    placeholder_to_dim_expr: dict[int, shape_poly._DimExpr]
    def __init__(self) -> None: ...
    def to_placeholder(self, dim_expr: Any) -> ir.Value: ...

DynamicShapeReplacementFn: Incomplete

@dataclasses.dataclass
class LoweringContext:
    grid_sizes: tuple[int, ...]
    grid_names: tuple[Hashable, ...] | None
    vmapped_dims: tuple[int, ...]
    user_grid_indices: Sequence[ir.Value] | None
    block_shapes: Sequence[tuple[int | pallas_core.Squeezed, ...] | None]
    name_stack: source_info_util.NameStack
    mesh_context: pallas_utils.MeshInfo | None
    kernel_type: tpu_core.CoreType
    traceback_caches: mlir.TracebackCaches
    forward_compatible: bool
    backend: xla_client.Client | None
    dynamic_shape_replacement_fn: DynamicShapeReplacementFn
    def replace(self, **changes: Any) -> LoweringContext: ...
    @property
    def grid_rank(self): ...
    @contextlib.contextmanager
    def grid_name_context(self) -> Generator[None]: ...

class ShapedAbstractValue(Protocol):
    shape: tuple[jax_core.DimSize, ...]
    dtype: jnp.dtype
    weak_type: bool
    def update(self, **kwargs: Any) -> Self: ...

@dataclasses.dataclass
class LoweringRuleContext:
    lowering_context: LoweringContext
    avals_in: Sequence[ShapedAbstractValue]
    avals_out: Sequence[ShapedAbstractValue]
    block_shapes: Sequence[tuple[int | pallas_core.Squeezed, ...] | None]
    def replace(self, **changes: Any) -> LoweringRuleContext: ...
    @property
    def forward_compatible(self): ...
    def is_cloud_tpu_older_than(self, year: int, month: int, day: int): ...
    def aval_to_ir_type(
        self,
        aval,
        *,
        shape=None,
        memory_space=None,
        is_kernel_boundary: bool = False,
        allow_extended_types: bool = True,
    ): ...

def aval_to_ir_type(
    dynamic_shape_replacement_fn: DynamicShapeReplacementFn,
    aval,
    *,
    shape=None,
    memory_space: AnyMemorySpace | None = None,
    is_kernel_boundary: bool = False,
    allow_extended_types: bool = True,
    kernel_type: tpu_core.CoreType,
): ...
def ir_constant(x: Any, mlir_type: ir.Type | None = None) -> ir.Value: ...

lowering_rules: Incomplete
skip_mlir_conversions: Incomplete
T = TypeVar("T")

def register_lowering_rule(
    prim: jax_core.Primitive,
    *,
    kernel_types: Collection[tpu_core.CoreType] = ...,
    ensure_mlir_values: bool = True,
) -> Callable[[T], T]: ...

@dataclasses.dataclass(init=False)
class MosaicGridMapping:
    grid: pallas_core.GridMappingGrid | None
    grid_names: tuple[Hashable, ...] | None
    jaxpr: jax_core.Jaxpr
    block_mappings: tuple[pallas_core.BlockMapping, ...]
    vmapped_dims: tuple[int, ...]
    scalar_prefetch_types: tuple[ir.Type, ...]
    operand_types: tuple[ir.Type, ...]
    scratch_types: tuple[ir.Type, ...]
    grid_types: tuple[ir.Type, ...]
    scalar_prefetch_block_shapes: tuple[tuple[int, ...], ...]
    operand_block_shapes: tuple[tuple[int | pallas_core.Squeezed, ...], ...]
    scratch_block_shapes: tuple[tuple[int, ...] | None, ...]
    mesh_info: pallas_utils.MeshInfo | None
    get_grid_indices: Callable[..., Any]
    def __init__(
        self,
        jaxpr: jax_core.Jaxpr,
        grid_mapping: pallas_core.GridMapping,
        dimension_semantics: Sequence[tpu_core.DimensionSemantics] | None,
        mesh: mesh_lib.Mesh | None,
        dynamic_shape_replacement_fn: DynamicShapeReplacementFn,
        kernel_type: tpu_core.CoreType,
    ) -> None: ...
    def maybe_compress_grid(self) -> None: ...
    @functools.cached_property
    def has_communication(self) -> bool: ...
    def get_dimension_semantics(self) -> ir.ArrayAttr: ...

def lower_jaxpr_to_module(
    lowering_context: mlir.LoweringRuleContext,
    grid_mapping: pallas_core.GridMapping,
    jaxpr: jax_core.Jaxpr,
    *,
    dimension_semantics: Sequence[tpu_core.DimensionSemantics] | None,
    kernel_type: tpu_core.CoreType,
    mesh: mesh_lib.Mesh | None = None,
    dynamic_shape_replacement_enabled: bool = False,
) -> ir.Module: ...
def lower_jaxpr_into_module(
    lowering_context: mlir.LoweringRuleContext,
    module: ir.Module,
    grid_mapping: pallas_core.GridMapping,
    jaxpr: jax_core.Jaxpr,
    *,
    name: str,
    dimension_semantics: Sequence[tpu_core.DimensionSemantics] | None,
    kernel_type: tpu_core.CoreType,
    mesh: mesh_lib.Mesh | None = None,
    dynamic_shape_replacement_enabled: bool = False,
) -> None: ...
def lower_jaxpr_to_transform_func(
    jaxpr: jax_core.Jaxpr,
    aval: jax_core.AbstractValue,
    *,
    name: str,
    mosaic_grid_mapping: MosaicGridMapping,
    kernel_type: tpu_core.CoreType,
    forward_compatible: bool,
    backend: Any | None,
    dynamic_shape_replacement_fn: DynamicShapeReplacementFn,
) -> func.FuncOp: ...
def lower_jaxpr_to_func(
    jaxpr: jax_core.Jaxpr,
    *,
    mosaic_grid_mapping: MosaicGridMapping,
    name: str,
    kernel_type: tpu_core.CoreType,
    forward_compatible: bool,
    backend: Any | None,
    dynamic_shape_replacement_fn: DynamicShapeReplacementFn,
    dynamic_shape_replacement_enabled: bool,
) -> func.FuncOp: ...
def lower_fun(fun: Callable, *, multiple_results: bool) -> Callable: ...

class LoweringException(Exception): ...

def jaxpr_subcomp(
    ctx: LoweringContext,
    jaxpr: jax_core.Jaxpr,
    *args: ir.Value,
) -> list[ir.Value]: ...

@dataclasses.dataclass(frozen=True)
class KeyScalarBundle:
    key_shape: tuple[int, ...]
    scalars: Sequence[ir.OpResult]

def reduce_lowering_rule(reduce_fn, type_to_kind, type_to_identity): ...

REDUCE_MAX_KINDS: Incomplete
REDUCE_MAX_IDENTITY: Incomplete
REDUCE_MIN_KINDS: Incomplete
REDUCE_MIN_IDENTITY: Incomplete
REDUCE_SUM_KINDS: Incomplete
REDUCE_SUM_IDENTITY: Incomplete

def jax_dot_dims_to_tpu_dot_dot_dims(dimension_numbers, lhs_shape, rhs_shape): ...

class FoldingError(Exception): ...

def random_seed_lowering(ctx: LoweringRuleContext, seeds, *, impl): ...
def random_bits_lowering(ctx: LoweringRuleContext, keys, *, bit_width, shape): ...
def random_fold_in_lowering(ctx: LoweringRuleContext, keys, msgs): ...
def random_unwrap_lowering(ctx: LoweringRuleContext, key): ...
def random_wrap_lowering(ctx: LoweringRuleContext, key_data, *, impl): ...
