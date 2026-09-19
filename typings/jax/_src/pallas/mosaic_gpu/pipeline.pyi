from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any, Protocol, TypeAlias, TypeVar

import dataclasses
import enum

from _typeshed import Incomplete
from jax import (
    api_util as api_util,
    lax as lax,
)
from jax._src import (
    core as core,
    state as state,
    util as util,
)
from jax._src.pallas import core as pallas_core
from jax.experimental import pallas as pl

import jax

map: Incomplete
zip: Incomplete
T = TypeVar("T")
BlockSpecPytree: TypeAlias
AbstractRefPytree: TypeAlias
map_brefs: Incomplete

@dataclasses.dataclass(frozen=True)
class BufferedRef:
    spec: pallas_core.BlockSpec = ...
    is_index_invariant: bool = ...
    gmem_ref: state.AbstractRef
    smem_ref: state.AbstractRef | None
    def get_ref_for_slot(self, slot: int | jax.Array) -> state.AbstractRef: ...
    def compute_gmem_slice(self, grid_indices) -> tuple[pl.Slice | jax.Array, ...]: ...
    def copy_in(self, slot, grid_indices, barrier_ref, barrier_slot=None) -> None: ...
    def copy_out(self, slot, grid_indices, predicate=None) -> None: ...

def emit_pipeline(
    body: Callable[..., T],
    *,
    grid: pallas_core.TupleGrid,
    in_specs: Sequence[pallas_core.BlockSpec] = (),
    out_specs: Sequence[pallas_core.BlockSpec] = (),
    max_concurrent_steps: int = 1,
    init_carry: T | None = None,
): ...

class ComputeContext(Protocol):
    def __call__(self, pipeline: Callable[[T], T]) -> None: ...

class PipelinePipeline(enum.IntEnum):
    START = 0
    STEADY = 1
    STOP = 2

class WarpSpecializedPipeline(Protocol):
    def __call__(self, *gmem_refs: Any, allocations: Any | None = None) -> None: ...
    def get_allocations(self, *gmem_refs: Any) -> Any: ...

def emit_pipeline_warp_specialized(
    body: Callable[..., None],
    *,
    grid: pallas_core.TupleGrid,
    memory_registers: int,
    in_specs: BlockSpecPytree = (),
    out_specs: BlockSpecPytree = (),
    max_concurrent_steps: int = 2,
    wg_axis: str,
    num_compute_wgs: int,
    pipeline_state: jax.Array | PipelinePipeline | None = None,
    manual_consumed_barriers: bool = False,
    compute_context: ComputeContext | None = None,
    memory_thread_idx: int | None = None,
) -> WarpSpecializedPipeline: ...
