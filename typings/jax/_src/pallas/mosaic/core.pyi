from collections.abc import Mapping, Sequence
from typing import Any, Literal

import abc
import dataclasses
import enum

from _typeshed import Incomplete
from jax._src import (
    core as jax_core,
    deprecations as deprecations,
    state as state,
    util as util,
)
from jax._src.frozen_dict import FrozenDict as FrozenDict
from jax._src.pallas import core as pallas_core

import jax
import jax.numpy as jnp
import numpy as np

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
no_block_spec: Incomplete

class CoreType(enum.Enum):
    TC = 0
    SC_SCALAR_SUBCORE = 1
    SC_VECTOR_SUBCORE = 2

class GridDimensionSemantics(enum.Enum):
    PARALLEL = "parallel"
    CORE_PARALLEL = "core_parallel"
    SUBCORE_PARALLEL = "subcore_parallel"
    ARBITRARY = "arbitrary"

PARALLEL: Incomplete
CORE_PARALLEL: Incomplete
SUBCORE_PARALLEL: Incomplete
ARBITRARY: Incomplete
DimensionSemantics: Incomplete

class SideEffectType(enum.Enum):
    PURE = "pure"
    DATAFLOW_SIDE_EFFECTING = "dataflow_side_effecting"
    SIDE_EFFECTING = "side_effecting"

@dataclasses.dataclass(frozen=True)
class CompilerParams:
    dimension_semantics: tuple[DimensionSemantics, ...] | None = ...
    allow_input_fusion: tuple[bool, ...] | None = ...
    vmem_limit_bytes: int | None = ...
    collective_id: int | None = ...
    has_side_effects: bool | SideEffectType = ...
    flags: dict[str, Any] | None = ...
    internal_scratch_in_bytes: int | None = ...
    serialization_format: int = ...
    kernel_type: CoreType = ...
    disable_bounds_checks: bool = ...
    disable_semaphore_checks: bool = ...
    skip_device_barrier: bool = ...
    allow_collective_id_without_custom_barrier: bool = ...
    shape_invariant_numerics: bool = ...
    use_tc_tiling_on_sc: bool | None = ...
    def __init__(
        self,
        dimension_semantics: Sequence[DimensionSemantics] | None = None,
        allow_input_fusion: Sequence[bool] | None = None,
        vmem_limit_bytes: int | None = None,
        collective_id: int | None = None,
        has_side_effects: bool | SideEffectType = False,
        flags: Mapping[str, Any] | None = None,
        internal_scratch_in_bytes: int | None = None,
        serialization_format: int = 1,
        kernel_type: CoreType = ...,
        disable_bounds_checks: bool = False,
        disable_semaphore_checks: bool = False,
        skip_device_barrier: bool = False,
        allow_collective_id_without_custom_barrier: bool = False,
        shape_invariant_numerics: bool = True,
        use_tc_tiling_on_sc: bool | None = None,
    ) -> None: ...
    replace = dataclasses.replace

class MemorySpace(enum.Enum):
    VMEM = "vmem"
    VMEM_SHARED = "vmem_shared"
    SMEM = "smem"
    CMEM = "cmem"
    SEMAPHORE = "semaphore_mem"
    HBM = "hbm"
    HOST = "host"
    def from_type(self, ty): ...
    def __call__(self, shape: Sequence[int], dtype: jnp.dtype[Any]): ...
    def __getattr__(self, name): ...

class dma_semaphore(pallas_core.semaphore_dtype, metaclass=abc.ABCMeta): ...

class DMASemaphore(pallas_core.AbstractSemaphoreTy):
    type = dma_semaphore
    name: str

class SemaphoreType(enum.Enum):
    REGULAR = "regular"
    DMA = "dma"
    BARRIER = "barrier"
    def __call__(self, shape: tuple[int, ...]): ...
    def get_array_aval(self) -> pallas_core.ShapedArrayWithMemorySpace: ...
    def get_ref_aval(self) -> state.AbstractRef: ...

@dataclasses.dataclass(frozen=True)
class AbstractSemaphore(jax_core.AbstractValue):
    sem_type: SemaphoreType

@dataclasses.dataclass(init=False, kw_only=True, unsafe_hash=True)
class PrefetchScalarGridSpec(pallas_core.GridSpec):
    num_scalar_prefetch: int
    def __init__(
        self,
        num_scalar_prefetch: int,
        grid: pallas_core.Grid = (),
        in_specs: pallas_core.BlockSpecTree = ...,
        out_specs: pallas_core.BlockSpecTree = ...,
        scratch_shapes: pallas_core.ScratchShapeTree = (),
    ) -> None: ...

@dataclasses.dataclass(frozen=True)
class TensorCore:
    id: int

@dataclasses.dataclass(frozen=True)
class TensorCoreMesh:
    devices: np.ndarray
    axis_names: Sequence[str]
    def __init__(self, devices: np.ndarray, axis_names: Sequence[str]) -> None: ...
    def __hash__(self) -> int: ...
    @property
    def kernel_type(self) -> CoreType: ...
    @property
    def default_memory_space(self) -> pallas_core.MemorySpace: ...
    @property
    def shape(self): ...
    @property
    def dimension_semantics(self) -> Sequence[str]: ...
    def discharges_effect(self, effect: jax_core.Effect) -> Literal[False]: ...

def create_tensorcore_mesh(
    axis_name: str,
    devices: Sequence[jax.Device] | None = None,
    num_cores: int | None = None,
) -> TensorCoreMesh: ...
def get_device_kind() -> str: ...
def get_num_device_cores() -> int: ...
