from collections.abc import (
    Callable as Callable,
    Hashable,
    Sequence,
)
from typing import Any

import enum

from _typeshed import Incomplete
from jax._src import (
    ad_util as ad_util,
    api_util as api_util,
    config as config,
    debugging as debugging,
    dtypes as dtypes,
    effects as effects,
    linear_util as lu,
    source_info_util as source_info_util,
    state as state,
    tree_util as tree_util,
    typing as jax_typing,
    util as util,
)
from jax._src.interpreters import ad as ad
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import arith as arith
from jax._src.pallas import (
    core as pallas_core,
    utils as pallas_utils,
)
from jax._src.state import indexing as indexing
from jax.interpreters import mlir as mlir

Slice: Incomplete
NDIndexer: Incomplete
map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
program_id_p: Incomplete

def program_id(axis: int) -> jax_typing.Array: ...
def program_id_bind_with_trace(trace, _, params): ...

num_programs_p: Incomplete

def num_programs(axis: int) -> int | jax_typing.Array: ...

class AtomicOpType(enum.Enum):
    XCHG = "xchg"
    ADD = "add"
    MAX = "max"
    MIN = "min"
    AND = "and"
    OR = "or"
    XOR = "xor"

atomic_rmw_p: Incomplete

def atomic_xchg(x_ref_or_view, idx, val, *, mask: Any | None = None): ...
def atomic_add(x_ref_or_view, idx, val, *, mask: Any | None = None): ...
def atomic_max(x_ref_or_view, idx, val, *, mask: Any | None = None): ...
def atomic_min(x_ref_or_view, idx, val, *, mask: Any | None = None): ...
def atomic_and(x_ref_or_view, idx, val, *, mask: Any | None = None): ...
def atomic_or(x_ref_or_view, idx, val, *, mask: Any | None = None): ...
def atomic_xor(x_ref_or_view, idx, val, *, mask: Any | None = None): ...

atomic_cas_p: Incomplete

def atomic_cas(ref, cmp, val): ...

max_contiguous_p: Incomplete

def max_contiguous(x, values): ...

multiple_of_p: Incomplete

def multiple_of(
    x: jax_typing.Array,
    values: Sequence[int] | int,
) -> jax_typing.Array: ...

load_p: Incomplete

def uninitialized_value(shape, dtype): ...

swap_p: Incomplete

def load(
    x_ref_or_view,
    idx,
    *,
    mask=None,
    other=None,
    cache_modifier=None,
    eviction_policy=None,
    volatile: bool = False,
) -> jax_typing.Array: ...
def swap(
    x_ref_or_view,
    idx,
    val,
    *,
    mask=None,
    eviction_policy=None,
    _function_name: str = "swap",
) -> jax_typing.Array: ...
def store(x_ref_or_view, idx, val, *, mask=None, eviction_policy=None) -> None: ...
def dot(
    a,
    b,
    trans_a: bool = False,
    trans_b: bool = False,
    allow_tf32: bool | None = None,
    precision=None,
): ...

reciprocal_p: Incomplete

def reciprocal(x, *, approx: bool = False): ...
def debug_print(fmt: str, *args: jax_typing.ArrayLike): ...
def check_debug_print_format(fmt: str, *args: jax_typing.ArrayLike): ...
@lu.transformation2
def wrap_with_transforms(f, transforms, *args): ...

run_scoped_p: Incomplete

def run_scoped(
    f: Callable[..., Any],
    *types: Any,
    collective_axes: Hashable | tuple[Hashable, ...] = (),
    **kw_types: Any,
) -> Any: ...

get_global_p: Incomplete

def get_global(what: pallas_core.ScratchShape) -> jax_typing.Array: ...

class DeviceIdType(enum.Enum):
    MESH = "mesh"
    LOGICAL = "logical"

def check_sem_avals(
    sem_aval,
    sem_transforms_avals,
    name,
    allowed_semaphore_types=None,
) -> None: ...

semaphore_read_p: Incomplete

def semaphore_read(sem_or_view) -> jax_typing.Array: ...

DeviceId: Incomplete

class SemaphoreEffect(effects.Effect): ...

sem_effect: Incomplete
semaphore_signal_p: Incomplete

def semaphore_signal(
    sem_or_view,
    inc: int | jax_typing.Array = 1,
    *,
    device_id: DeviceId = None,
    device_id_type: DeviceIdType = ...,
    core_index: int | jax_typing.Array | None = None,
): ...

semaphore_wait_p: Incomplete

def semaphore_wait(
    sem_or_view,
    value: int | jax_typing.Array = 1,
    *,
    decrement: bool = True,
): ...
def device_id_to_logical(
    mesh_context: pallas_utils.MeshInfo | None,
    device_id: ir.Value | tuple[ir.Value, ...] | dict[Any, ir.Value],
    device_id_type: DeviceIdType,
    get_axis_index,
) -> tuple[ir.Value | None, dict[Any, ir.Value]]: ...

delay_p: Incomplete

class DelayEffect(effects.Effect): ...

delay_effect: Incomplete

def delay(nanos: int | jax_typing.Array) -> None: ...
