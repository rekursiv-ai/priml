from collections.abc import Sequence
from typing import Any

import dataclasses

from jax._src import (
    core as jax_core,
    state as state,
    tree_util as tree_util,
)
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic import (
    core as tpu_core,
    tpu_info as tpu_info,
)

import jax

type Tiling = Sequence[Sequence[int]]

@dataclasses.dataclass(frozen=True)
class MemoryRef(pallas_core.MemoryRef):
    tiling: Tiling | None = ...
    def __init__(
        self,
        shape: Sequence[int],
        dtype: jax.typing.DTypeLike,
        memory_space: tpu_core.MemorySpace,
        tiling: Tiling | None = None,
    ) -> None: ...
    def get_ref_aval(self) -> state.TransformedRef | state.AbstractRef: ...

class AbstractRef(state.AbstractRef):
    tiling: Tiling | None
    def __init__(
        self,
        aval: jax_core.AbstractValue,
        memory_space: tpu_core.MemorySpace,
        tiling: Tiling | None,
    ) -> None: ...
    def update(
        self,
        inner_aval: Any | None = None,
        memory_space: Any | None = None,
        tiling: Tiling | None = None,
    ) -> AbstractRef: ...

@dataclasses.dataclass
class BlockSpec(pallas_core.BlockSpec):
    indexed_by: int | None = ...
    indexed_dim: int | None = ...
    def __post_init__(self) -> None: ...
    def to_block_mapping(
        self,
        origin: pallas_core.OriginStr,
        array_aval: jax_core.ShapedArray,
        *,
        index_map_avals: Sequence[jax_core.AbstractValue],
        index_map_tree: tree_util.PyTreeDef,
        grid: pallas_core.GridMappingGrid,
        vmapped_dims: tuple[int, ...],
        debug: bool = False,
    ) -> BlockMapping: ...

@dataclasses.dataclass(frozen=True)
class BlockMapping(pallas_core.BlockMapping):
    indexed_by: int | None = ...
    indexed_dim: int | None = ...

def get_sparse_core_info() -> tpu_info.SparseCoreInfo: ...

@dataclasses.dataclass(frozen=True, kw_only=True)
class ScalarSubcoreMesh:
    axis_name: str
    num_cores: int
    @property
    def kernel_type(self) -> tpu_core.CoreType: ...
    @property
    def default_memory_space(self) -> tpu_core.MemorySpace: ...
    @property
    def shape(self): ...
    @property
    def dimension_semantics(self) -> Sequence[str]: ...
    def discharges_effect(self, effect): ...

@dataclasses.dataclass(frozen=True, kw_only=True)
class VectorSubcoreMesh:
    core_axis_name: str
    subcore_axis_name: str
    num_cores: int = ...
    num_subcores: int = ...
    def __post_init__(self) -> None: ...
    @property
    def kernel_type(self) -> tpu_core.CoreType: ...
    @property
    def default_memory_space(self) -> tpu_core.MemorySpace: ...
    @property
    def shape(self): ...
    @property
    def dimension_semantics(self) -> Sequence[str]: ...
    def discharges_effect(self, effect): ...

def supported_shapes(dtype: jax.typing.DTypeLike) -> Sequence[tuple[int, ...]]: ...
