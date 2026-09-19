from collections.abc import (
    Callable as Callable,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Sequence,
    Set as AbstractSet,
)
from typing import Any, ClassVar, Protocol

import abc
import collections
import contextlib
import dataclasses
import enum
import functools
import threading

from _typeshed import Incomplete
from jax._src import (
    api_util as api_util,
    config as config,
    core as jax_core,
    dtypes as dtypes,
    effects as effects,
    frozen_dict as frozen_dict,
    hijax as hijax,
    numpy as jnp,
    state as state,
    tree_util as tree_util,
    typing as jax_typing,
    util as util,
)
from jax._src.api import jit as jit
from jax._src.export._export import export as export
from jax._src.interpreters import mlir as mlir
from jax._src.state import (
    indexing as indexing,
    types as state_types,
)
from jax._src.state.types import TransformedRef as TransformedRef

class DynamicGridDim: ...

dynamic_grid_dim: Incomplete
partial = functools.partial
GridElement: Incomplete
GridName = Hashable
type GridNames = tuple[Hashable, ...] | None
type NamedGrid = tuple[tuple[GridName, int], ...]
type TupleGrid = tuple[GridElement, ...]
type Grid = NamedGrid | TupleGrid
type StaticGrid = tuple[int, ...]
type GridMappingGrid = tuple[int | DynamicGridDim, ...]
OriginStr = str
SEMAPHORE_INTERPRET_DTYPE: Incomplete
SEMAPHORE_MAX_VALUE: Incomplete

class AbstractSemaphoreTyRules:
    @staticmethod
    def pallas_interpret_element_aval(_) -> jax_core.ShapedArray: ...
    @staticmethod
    def physical_element_aval(_) -> jax_core.ShapedArray: ...

class AbstractSemaphoreTy(dtypes.ExtendedDType, metaclass=abc.ABCMeta):
    name: str
    def __eq__(self, other): ...
    def __hash__(self) -> int: ...

class semaphore_dtype(dtypes.extended, metaclass=abc.ABCMeta): ...
class semaphore(semaphore_dtype, metaclass=abc.ABCMeta): ...

class Semaphore(AbstractSemaphoreTy):
    name: str
    type = semaphore

class barrier_semaphore(semaphore_dtype, metaclass=abc.ABCMeta): ...

class BarrierSemaphore(AbstractSemaphoreTy):
    name: str
    type = barrier_semaphore

class CompilerParams(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]

@dataclasses.dataclass(frozen=True)
class Buffered:
    buffer_count: int
    use_lookahead: bool = ...

split_list: Incomplete
map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete

class ShapedArrayWithMemorySpace(jax_core.ShapedArray):
    memory_space: Incomplete
    def __init__(
        self,
        shape,
        dtype,
        weak_type: bool = False,
        sharding=None,
        vma=...,
        memory_space=None,
    ) -> None: ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def str_short(self, short_dtypes: bool = False, mesh_axis_types: bool = False): ...
    def update(
        self,
        shape=None,
        dtype=None,
        weak_type=None,
        sharding=None,
        vma=None,
        memory_space=None,
    ): ...
    def unwrap(self) -> jax_core.ShapedArray: ...

@dataclasses.dataclass(frozen=True)
class MemoryRef:
    inner_aval: jax_core.AbstractValue
    memory_space: Any
    def get_array_aval(self) -> jax_core.ShapedArray: ...
    def get_ref_aval(self) -> TransformedRef | state.AbstractRef: ...
    @property
    def dtype(self): ...
    @property
    def shape(self): ...
    def __lt__(self, other): ...

class MemorySpace(enum.Enum):
    ANY = "any"
    ERROR = "error"
    INDEX = "index"
    KEY = "key"
    HOST = "host"
    def from_type(self, type: jax_core.AbstractValue) -> MemoryRef: ...
    def __call__(self, shape: tuple[int, ...], dtype: jnp.dtype): ...

@dataclasses.dataclass(frozen=True)
class PallasGridContext:
    grid: GridMappingGrid
    mapped_dims: tuple[int, ...]
    def size(self, axis: int) -> int | DynamicGridDim: ...

@dataclasses.dataclass
class PallasTracingEnv(threading.local):
    grid_context: PallasGridContext | None = ...
    grid_env_stack: list[GridEnv] = ...
    is_interpret_mode: bool = ...
    dynamic_shapes: bool = ...
    module_export_fn: Callable[[mlir.ir.Module], None] | None = ...

def axis_frame() -> PallasGridContext: ...

@dataclasses.dataclass(frozen=True)
class GridAxis:
    index: jax_typing.Array
    size: int

type GridEnv = Sequence[GridAxis]

@contextlib.contextmanager
def grid_env(env: GridEnv) -> Iterator[None]: ...
def current_grid_env() -> GridEnv | None: ...

@dataclasses.dataclass(frozen=True)
class Element:
    block_size: int
    padding: tuple[int, int] = ...

@dataclasses.dataclass(frozen=True)
class Squeezed: ...

squeezed: Incomplete

@dataclasses.dataclass(frozen=True)
class Blocked:
    block_size: int

@dataclasses.dataclass(frozen=True)
class BoundedSlice:
    block_size: int

type BlockDim = Element | Squeezed | Blocked | BoundedSlice

def default_index_map(ndim: int) -> Callable: ...
def get_block_size(dim: BlockDim | int | None) -> int: ...

class _IndexMapFunc:
    index_map: Incomplete
    def __init__(self, index_map) -> None: ...
    def __eq__(self, other: object): ...
    def __call__(self, *args, **kwargs): ...

@dataclasses.dataclass
class BlockSpec:
    block_shape: Sequence[BlockDim | int | None] | None = ...
    index_map: Callable[..., Any] | None = ...
    memory_space: Any | None = ...
    pipeline_mode: Buffered | None = ...
    def __post_init__(self) -> None: ...
    def to_block_mapping(
        self,
        origin: OriginStr,
        array_aval: jax_core.ShapedArray,
        *,
        index_map_avals: Sequence[jax_core.AbstractValue],
        index_map_tree: tree_util.PyTreeDef,
        grid: GridMappingGrid,
        vmapped_dims: tuple[int, ...],
        debug: bool = False,
    ) -> BlockMapping: ...
    replace = dataclasses.replace

class NoBlockSpec: ...

no_block_spec: Incomplete
type BlockSpecTree = Any

def undo_transforms(
    aval: jax_core.AbstractValue,
    memory_transforms: Sequence[state_types.Transform],
) -> list[state_types.Transform]: ...

@dataclasses.dataclass(frozen=True)
class BlockMapping:
    block_shape: tuple[BlockDim, ...]
    transformed_block_aval: state.AbstractRef
    index_map_jaxpr: jax_core.ClosedJaxpr
    index_map_out_tree: tree_util.PyTreeDef
    array_aval: jax_core.ShapedArray
    origin: OriginStr
    transforms: Sequence[state_types.Transform] = ...
    pipeline_mode: Buffered | None = ...
    debug: bool = ...
    def check_invariants(self) -> None: ...
    def replace(self, **kwargs): ...
    @property
    def block_aval(self) -> state.AbstractRef: ...
    @property
    def ref_aval(self) -> state.AbstractRef | TransformedRef: ...
    def compute_start_indices_interpret(self, loop_idx, *args): ...
    def has_trivial_window(self): ...
    def to_block_spec(self) -> BlockSpec: ...
    def to_lojax(
        self,
        index_map_avals,
        index_map_tree,
        grid,
        vmapped_dims,
    ) -> list[BlockMapping]: ...

@contextlib.contextmanager
def tracing_grid_env(grid: GridMappingGrid, mapped_dims: tuple[int, ...]): ...
@contextlib.contextmanager
def pallas_export_experimental(dynamic_shapes: bool): ...
def dynamic_shapes_export_enabled() -> bool: ...
def is_dynamic_dim(d) -> bool: ...

@dataclasses.dataclass(frozen=True)
class GridMapping:
    grid: GridMappingGrid
    grid_names: tuple[Hashable, ...] | None
    block_mappings: tuple[BlockMapping, ...]
    index_map_tree: tree_util.PyTreeDef
    index_map_avals: tuple[jax_core.AbstractValue, ...]
    vmapped_dims: tuple[int, ...]
    scratch_avals: tuple[jax_core.AbstractValue, ...]
    num_index_operands: int
    num_inputs: int
    num_outputs: int
    get_grid_indices: Callable | None = ...
    local_grid_env: Callable | None = ...
    debug: bool = ...
    def check_invariants(self) -> None: ...
    def replace(self, **kwargs) -> GridMapping: ...
    @property
    def num_dynamic_grid_bounds(self): ...
    @property
    def num_scratch_operands(self): ...
    @property
    def static_grid(self) -> StaticGrid: ...
    @contextlib.contextmanager
    def trace_env(self) -> Generator[None]: ...
    @property
    def slice_index_ops(self): ...
    @property
    def slice_block_ops(self): ...
    @property
    def slice_scratch_ops(self): ...
    @property
    def in_shapes(self) -> Iterable[jax_core.ShapeDtypeStruct]: ...
    @property
    def block_mappings_output(self) -> Iterable[BlockMapping]: ...
    @property
    def out_shapes(self) -> Iterable[jax_core.ShapeDtypeStruct]: ...
    def to_lojax(self): ...

index_map_grid_aval: Incomplete

class ScratchShape(Protocol):
    def get_array_aval(self) -> jax_core.AbstractValue: ...
    def get_ref_aval(self) -> state.AbstractRef | TransformedRef: ...

ScratchShapeTree: Incomplete

@dataclasses.dataclass(init=False, kw_only=True)
class GridSpec:
    grid: TupleGrid
    grid_names: tuple[Hashable, ...] | None
    in_specs: BlockSpecTree
    out_specs: BlockSpecTree
    scratch_shapes: ScratchShapeTree = ...
    def __init__(
        self,
        grid: Grid = (),
        in_specs: BlockSpecTree = ...,
        out_specs: BlockSpecTree = ...,
        scratch_shapes: ScratchShapeTree = (),
    ) -> None: ...

def get_grid_mapping(
    grid_spec: GridSpec,
    in_avals: Sequence[jax_core.AbstractValue],
    in_tree: tree_util.PyTreeDef,
    in_origins: Sequence[OriginStr],
    out_avals: Sequence[jax_core.AbstractValue],
    out_tree: tree_util.PyTreeDef,
    out_origins: Sequence[OriginStr],
    debug: bool = False,
) -> tuple[tuple[jax_core.AbstractValue, ...], GridMapping]: ...
def unzip_dynamic_grid_bounds(
    grid_spec: GridSpec,
) -> tuple[GridSpec, tuple[Any, ...]]: ...
def pytreedef_mismatch_err_msg(
    what1: str,
    tree1: tree_util.PyTreeDef,
    what2: str,
    tree2: tree_util.PyTreeDef,
) -> str: ...

@dataclasses.dataclass(frozen=True)
class CostEstimate:
    flops: int
    transcendentals: int
    bytes_accessed: int
    remote_bytes_transferred: int = ...
    def __post_init__(self) -> None: ...
    def to_json(self) -> bytes: ...

def get_memory_space_aval(aval: jax_core.AbstractValue) -> Any: ...

core_map_p: Incomplete

def core_map(
    mesh,
    *,
    compiler_params: Any | None = None,
    interpret: bool = False,
    debug: bool = False,
    cost_estimate: CostEstimate | None = None,
    name: str | None = None,
    metadata: dict[str, str] | None = None,
    scratch_shapes: ScratchShapeTree = (),
): ...

class CommsEffect(effects.Effect): ...

comms_effect: Incomplete
kernel_local_effects: effects.EffectTypeSet

def get_interpret_effects(interpret: Any) -> AbstractSet[effects.Effect]: ...
def core_map_lowering_rule(ctx: mlir.LoweringRuleContext, *args, jaxpr, **kwargs): ...

class Mesh(Protocol):
    @property
    def default_memory_space(self) -> MemorySpace | Any: ...
    @property
    def shape(self) -> collections.OrderedDict[object, int]: ...
    def discharges_effect(self, effect: jax_core.Effect) -> bool: ...

with_memory_space_constraint_p: Incomplete

@with_memory_space_constraint_p.def_impl
def with_memory_space_constraint_impl(x, *, memory_space) -> None: ...
@with_memory_space_constraint_p.def_abstract_eval
def with_memory_space_constraint_abstract_eval(x, *, memory_space): ...
def with_memory_space_constraint_lowering_rule(ctx, x, *, memory_space): ...
def default_mesh_discharge_rule(
    in_avals,
    out_avals,
    *args,
    mesh,
    compiler_params,
    jaxpr,
    debug,
    interpret,
    cost_estimate,
    name,
    metadata,
): ...
def lower_as_mlir(
    f,
    *args,
    dynamic_shapes: bool = False,
    device=None,
    static_argnames=(),
    platforms=None,
    **kwargs,
) -> mlir.ir.Module: ...
