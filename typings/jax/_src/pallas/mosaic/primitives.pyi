from typing import Any

import dataclasses

from _typeshed import Incomplete
from jax._src import (
    dtypes as dtypes,
    effects as effects,
    state as state,
    tree_util as tree_util,
    util as util,
)
from jax._src.interpreters import mlir as mlir
from jax._src.pallas import primitives as primitives
from jax._src.state import indexing as indexing
from jax._src.state.types import Transform as Transform
from jax._src.typing import DTypeLike as DTypeLike

import jax

Slice: Incomplete
map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
IntDeviceId: Incomplete
type MultiDimDeviceId = (
    tuple[IntDeviceId, ...] | dict[str | tuple[str, ...], IntDeviceId]
)
Ref: Incomplete

def repeat(x: jax.Array, repeats: int, axis: int) -> jax.Array: ...

bitcast_p: Incomplete

def bitcast(x: jax.Array, ty: DTypeLike) -> jax.Array: ...

roll_p: Incomplete

def roll(
    x: jax.Array,
    shift: jax.Array | int,
    axis: int,
    *,
    stride: int | None = None,
    stride_axis: int | None = None,
) -> jax.Array: ...

@dataclasses.dataclass
class AsyncCopyDescriptor:
    src_ref: Any
    src_transforms: tuple[Transform, ...]
    dst_ref: Any
    dst_transforms: tuple[Transform, ...]
    dst_sem: int | jax.Array
    dst_sem_transforms: tuple[Transform, ...]
    src_sem: int | jax.Array | None
    src_sem_transforms: tuple[Transform, ...] | None
    device_id: MultiDimDeviceId | IntDeviceId | None
    device_id_type: primitives.DeviceIdType = ...
    def __post_init__(self) -> None: ...
    def __del__(self) -> None: ...
    @property
    def is_remote(self): ...
    def start(self, priority: int = 0, *, add: bool = False): ...
    def wait(self) -> None: ...
    def wait_recv(self) -> None: ...
    def wait_send(self) -> None: ...

@dataclasses.dataclass(frozen=True)
class TransformedRefTree(state.TransformedRef):
    @classmethod
    def wrap(cls, ref: state.TransformedRef) -> TransformedRefTree: ...
    def unwrap(self) -> state.TransformedRef: ...

dma_start_p: Incomplete

def dma_start_partial_discharge_rule(
    should_discharge,
    in_avals,
    out_avals,
    *args,
    tree,
    device_id_type,
    priority,
    add,
): ...

dma_wait_p: Incomplete

def dma_wait_partial_discharge_rule(
    should_discharge,
    in_avals,
    out_avals,
    *args,
    tree,
    device_id_type,
): ...
def make_async_copy(src_ref, dst_ref, sem) -> AsyncCopyDescriptor: ...
def async_copy(
    src_ref,
    dst_ref,
    sem,
    *,
    priority: int = 0,
    add: bool = False,
) -> AsyncCopyDescriptor: ...
def make_async_remote_copy(
    src_ref,
    dst_ref,
    send_sem,
    recv_sem,
    device_id: MultiDimDeviceId | IntDeviceId | None,
    device_id_type: primitives.DeviceIdType = ...,
) -> AsyncCopyDescriptor: ...
def async_remote_copy(
    src_ref,
    dst_ref,
    send_sem,
    recv_sem,
    device_id,
    device_id_type: primitives.DeviceIdType = ...,
) -> AsyncCopyDescriptor: ...

get_barrier_semaphore_p: Incomplete

def get_barrier_semaphore(): ...

prng_seed_p: Incomplete

class PRNGEffect(effects.Effect): ...

prng_effect: Incomplete

def prng_seed(*seeds: int | jax.Array) -> None: ...

prng_random_bits_p: Incomplete

def prng_random_bits(shape): ...

split_key_p: Incomplete

def unwrap_pallas_seed(seed): ...

join_key_p: Incomplete

def wrap_pallas_seed(*seeds, impl): ...

stochastic_round_p: Incomplete

def stochastic_round(x, random_bits, *, target_dtype): ...

pack_elementwise_p: Incomplete

def pack_elementwise(xs, *, packed_dtype): ...

unpack_elementwise_p: Incomplete

def unpack_elementwise(x, *, index, packed_dtype, unpacked_dtype): ...
def with_memory_space_constraint(x: jax.Array, memory_space: Any) -> jax.Array: ...
def load(ref: Ref, *, mask: jax.Array | None = None) -> jax.Array: ...
def store(ref: Ref, val: jax.Array, *, mask: jax.Array | None = None) -> None: ...

touch_p: Incomplete

def touch(ref: jax.Array | state.TransformedRef) -> None: ...

trace_value_p: Incomplete

def trace_value(label: str, value: jax.Array) -> None: ...

class TraceEffect(effects.Effect): ...

trace_effect: Incomplete

class MXUEffect(effects.Effect): ...

mxu_effect: Incomplete
matmul_push_rhs_p: Incomplete

def matmul_push_rhs(rhs: jax.Array, staging_register: int, mxu_index: int) -> None: ...

matmul_acc_lhs_p: Incomplete

def matmul_acc_lhs(
    acc_addr: int,
    lhs: jax.Array,
    mxu_index: int,
    load_staged_rhs: int | None = None,
) -> None: ...

matmul_pop_p: Incomplete

def matmul_pop(
    acc_addr: int,
    shape: tuple[int, int],
    dtype: jax.typing.DTypeLike,
    mxu_index: int,
): ...
