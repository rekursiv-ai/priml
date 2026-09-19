from collections.abc import Mapping
from typing import Self

import dataclasses

from jax import numpy as jnp
from jax._src import (
    callback as callback,
    source_info_util as source_info_util,
)
from jax._src.pallas.mosaic.interpret import utils as interpret_utils
from jax._src.pallas.mosaic.interpret.race_detection_state import (
    RaceDetectionState as RaceDetectionState,
)
from jax._src.pallas.mosaic_gpu import core as mosaic_gpu_core
from jax._src.state import indexing as indexing

import jax
import numpy as np

IDX_BY_GPU_MEMORY_SPACE: Mapping[mosaic_gpu_core.MemorySpace, int]
GPU_MEMORY_SPACE_BY_IDX: Mapping[int, mosaic_gpu_core.MemorySpace]

def get_memory_space_idx(space: mosaic_gpu_core.MemorySpace | None) -> int: ...
def is_smem_memory_space(space: mosaic_gpu_core.MemorySpace | None) -> bool: ...
def is_gmem_memory_space(space: mosaic_gpu_core.MemorySpace | None) -> bool: ...
def get_races() -> RaceDetectionState: ...
def reset_gpu_interpret_mode_state() -> None: ...
def call_initialize_shared_memory(
    *,
    num_devices: int,
    num_threads: int,
    interpret_params: interpret_utils.InterpretGPUParams,
): ...
def call_clean_up_shared_memory() -> None: ...
def call_update_clocks_for_device_barrier(device_id: int): ...

@dataclasses.dataclass(frozen=True, kw_only=True)
class HostAllocationRequest:
    memory_space_id: int
    device_id: int
    thread_id: int = ...
    initial_ref_count: int = ...
    def __iter__(self): ...
    @classmethod
    def shape_and_dtype(cls) -> jax.ShapeDtypeStruct: ...
    @property
    def as_array(self) -> np.ndarray: ...
    @property
    def as_jax_array(self) -> jnp.ndarray: ...
    @classmethod
    def from_array(cls, request: np.ndarray | jnp.ndarray) -> Self: ...

def make_allocation_request_array(
    *,
    memory_space_id: int,
    device_id: int,
    thread_id: int = 0,
    initial_ref_count: int = 1,
) -> jnp.ndarray: ...

@dataclasses.dataclass(frozen=True, kw_only=True)
class HostAllocationKey(HostAllocationRequest):
    buffer_id: int
    def __iter__(self): ...

def call_allocate_buffer_for_all_threads(
    device_id: int,
    allocation_request: jnp.ndarray,
    value: jnp.ndarray,
) -> jnp.ndarray: ...
def call_allocate_buffer(
    device_id: int,
    thread_id: int,
    allocation_request: jnp.ndarray,
    value: jnp.ndarray,
) -> jnp.ndarray: ...
def call_deallocate_buffer(allocation_key: jnp.ndarray): ...
def call_get(
    *,
    result_shape_and_dtype,
    device_id: int,
    thread_id: int,
    allocation_key: jnp.ndarray,
    transforms,
    block_indices=None,
    grid_loop_idx=None,
    clock=None,
    source_info=None,
    input_name=None,
) -> jnp.ndarray: ...
def call_swap(
    *,
    result_shape_and_dtype,
    device_id: int,
    thread_id: int,
    allocation_key: jnp.ndarray,
    transforms,
    val,
    mask,
    source_info=None,
): ...
def call_allocate_barriers(
    device_id: int,
    thread_id: int,
    num_arrivals: int,
    num_barriers: int,
    ref_count: int,
) -> jnp.ndarray: ...
def call_deallocate_barrier(
    device_id: int,
    thread_id: int,
    allocation_key: jnp.ndarray,
): ...
def call_barrier_wait(device_id: int, thread_id: int, allocation_key: jnp.ndarray): ...
def call_barrier_arrive(
    device_id: int,
    thread_id: int,
    allocation_key: jnp.ndarray,
): ...
def call_assert_no_barriers_allocated() -> None: ...
