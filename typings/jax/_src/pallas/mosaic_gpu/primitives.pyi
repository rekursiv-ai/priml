from collections.abc import (
    Callable as Callable,
    Hashable,
    Sequence,
)
from typing import Literal

import dataclasses

from _typeshed import Incomplete
from jax._src import (
    core as jax_core,
    debugging as debugging,
    dtypes as dtypes,
    literals as literals,
    state as state,
    tree_util as tree_util,
    util as util,
)
from jax._src.lib.mlir import ir as ir
from jax._src.pallas import primitives as pallas_primitives
from jax._src.pallas.mosaic_gpu import (
    core as gpu_core,
    lowering as lowering,
)
from jax._src.pallas.mosaic_gpu.core import state_types as state_types
from jax._src.state import (
    discharge as discharge,
    indexing as indexing,
)
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.mosaic.gpu import tcgen05 as tcgen05

import jax
import jax.numpy as jnp

AxisName: Incomplete
WARP_SIZE: int
WARPGROUP_SIZE: int
SomeLayout: Incomplete
print_layout_p: Incomplete

def print_layout(fmt: str, x: jax.typing.ArrayLike | _Ref) -> None: ...

copy_smem_to_gmem_p: Incomplete

def copy_smem_to_gmem(
    src: _Ref,
    dst: _Ref,
    predicate: jax.Array | None = None,
    *,
    commit_group: bool = True,
    reduction_op: mgpu.TMAReductionOp | None = None,
) -> None: ...

copy_gmem_to_smem_p: Incomplete

def copy_gmem_to_smem(
    src: _Ref,
    dst: _Ref,
    barrier: _Ref,
    *,
    collective_axes: str | tuple[str, ...] | None = None,
    partitioned_axis: int | None = None,
) -> None: ...

async_prefetch_p: Incomplete

def async_prefetch(
    ref: _Ref,
    *,
    collective_axes: str | tuple[str, ...] | None = None,
    partitioned_axis: int | None = None,
) -> None: ...

barrier_arrive_p: Incomplete

def barrier_arrive(barrier: state.AbstractRef) -> None: ...

barrier_wait_p: Incomplete

def barrier_wait(barrier: state.AbstractRef) -> None: ...

wait_smem_to_gmem_p: Incomplete

def wait_smem_to_gmem(n: int, wait_read_only: bool = False) -> None: ...

commit_group_p: Incomplete

def commit_smem_to_gmem_group() -> None: ...

wgmma_ref_p: Incomplete

def wgmma(acc: gpu_core.WGMMAAbstractAccumulatorRef, a, b) -> None: ...

wgmma_p: Incomplete
wgmma_wait_p: Incomplete

def wgmma_wait(n: int): ...
@wgmma_wait_p.def_effectful_abstract_eval
def wgmma_wait_effectful_abstract_eval(_): ...

wgmma_accumulator_deref_p: Incomplete

def wgmma_accumulator_load(acc, *, wait_n: int | None = 0): ...

wgmma_accumulator_store_p: Incomplete

def wgmma_accumulator_store(acc_ref, val) -> None: ...

tcgen05_mma_p: Incomplete

def tcgen05_mma(
    acc: _Ref,
    a: _Ref,
    b: _Ref,
    barrier: _Ref | None = None,
    *,
    a_scale: _Ref | None = None,
    b_scale: _Ref | None = None,
    a_sparse_metadata: _Ref | None = None,
    accumulate: bool | jax.Array = True,
    collective_axis: str | None = None,
): ...

tcgen05_commit_arrive_p: Incomplete

def tcgen05_commit_arrive(barrier: _Ref, collective_axis: str | None = None): ...

commit_tmem_p: Incomplete

def commit_tmem() -> None: ...

set_max_registers_p: Incomplete

def set_max_registers(n: int, *, action: Literal["increase", "decrease"]): ...

commit_smem_p: Incomplete

def commit_smem() -> None: ...
def broadcasted_iota(
    dtype: jax.typing.DTypeLike,
    shape: Sequence[int],
    dimension: int,
    *,
    layout: SomeLayout | None = None,
) -> jax.Array: ...

jaxpr_call_p: Incomplete

def jaxpr_call(
    jaxpr: jax_core.Jaxpr,
    *refs: state.AbstractRef | state_types.TransformedRef,
    program_ids: Sequence[jax.Array | None],
) -> Sequence[jax.Array]: ...

@dataclasses.dataclass(frozen=True)
class ShapeDtypeStruct:
    shape: tuple[int, ...]
    dtype: jnp.dtype
    layout: SomeLayout

inline_mgpu_p: Incomplete

@dataclasses.dataclass(frozen=True)
class RefType:
    transforms: tuple[state_types.Transform, ...] = ...

def inline_mgpu(*, arg_types=(), return_type=None): ...

load_p: Incomplete

def load(
    src: _Ref,
    idx,
    *,
    layout: SomeLayout | None = None,
    optimized: bool = True,
) -> jax.Array: ...

async_load_tmem_p: Incomplete

def async_load_tmem(src: _Ref, *, layout: SomeLayout | None = None) -> jax.Array: ...

wait_load_tmem_p: Incomplete

def wait_load_tmem() -> None: ...

async_store_tmem_p: Incomplete

def async_store_tmem(ref: _Ref, value): ...

async_copy_scales_to_tmem_p: Incomplete

def async_copy_scales_to_tmem(
    smem_ref: _Ref,
    tmem_ref: _Ref,
    collective_axis: AxisName | None = None,
): ...

async_copy_sparse_metadata_to_tmem_p: Incomplete

def async_copy_sparse_metadata_to_tmem(
    smem_ref: _Ref,
    tmem_ref: _Ref,
    collective_axis: AxisName | None = None,
): ...

async_copy_smem_to_tmem_p: Incomplete

def async_copy_smem_to_tmem(
    smem_ref: _Ref,
    tmem_ref: _Ref,
    collective_axis: AxisName | None = None,
): ...

semaphore_signal_parallel_p: Incomplete

@dataclasses.dataclass(frozen=True)
class SemaphoreSignal:
    ref: _Ref
    _: dataclasses.KW_ONLY
    device_id: pallas_primitives.DeviceId | None
    inc: int | jax.Array = ...

def semaphore_signal_parallel(*signals: SemaphoreSignal): ...

try_cluster_cancel_p: Incomplete

def try_cluster_cancel_lowering(
    ctx: lowering.LoweringRuleContext,
    result_ref,
    barrier: mgpu.BarrierRef,
    *transforms_leaves,
    result_transforms_tree,
    barrier_transforms_tree,
): ...
def try_cluster_cancel(result_ref: _Ref, barrier: _Ref) -> None: ...

query_cluster_cancel_p: Incomplete

def query_cluster_cancel_lowering(
    ctx: lowering.LoweringRuleContext,
    result_ref,
    *transforms_leaves,
    grid_names,
    transforms_tree,
): ...
def query_cluster_cancel(
    result_ref: _Ref,
    grid_names: Sequence[Hashable],
) -> tuple[tuple[jax.Array, ...], jax.Array]: ...

multimem_store_p: Incomplete

def multimem_store(
    source: jax.Array,
    ref: _Ref,
    collective_axes: Hashable | tuple[Hashable, ...],
): ...

multimem_load_reduce_p: Incomplete

def multimem_load_reduce(
    ref: _Ref,
    *,
    collective_axes: Hashable | tuple[Hashable, ...],
    reduction_op: mgpu.MultimemReductionOp,
) -> jax.Array: ...

semaphore_signal_multicast_p: Incomplete

def semaphore_signal_multicast(
    semaphore,
    value: int | jax.Array = 1,
    *,
    collective_axes: Hashable | tuple[Hashable, ...],
): ...
