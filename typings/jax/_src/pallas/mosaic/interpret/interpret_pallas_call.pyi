from collections.abc import Callable as Callable
from typing import Any, Literal

import contextlib
import dataclasses
import enum
import threading

from _typeshed import Incomplete
from jax import lax as lax
from jax._src import (
    callback as callback,
    config as config,
    core as jax_core,
    frozen_dict as frozen_dict,
    pjit as pjit,
    source_info_util as source_info_util,
    state as state,
)
from jax._src.interpreters import mlir as mlir
from jax._src.pallas import (
    core as pallas_core,
    primitives as primitives,
)
from jax._src.pallas.mosaic import core as mosaic_core
from jax._src.pallas.mosaic.interpret import (
    shared_memory as memory,
    vector_clock as vc,
)
from jax._src.pallas.mosaic.interpret.race_detection_state import (
    RaceDetectionState as RaceDetectionState,
)
from jax._src.pallas.mosaic.interpret.thread_map import thread_map as thread_map
from jax._src.state import indexing as indexing
from jax._src.tree_util import FlatTree as FlatTree
from jax._src.typing import Array as Array
from jax._src.util import (
    safe_map as safe_map,
    safe_zip as safe_zip,
    split_list as split_list,
)

import jax._src.pallas.mosaic.interpret.utils as interpret_utils
import numpy as np

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete

@dataclasses.dataclass(frozen=True, kw_only=True)
class InterpretParams(interpret_utils.InterpretParams):
    dma_execution_mode: Literal["eager", "on_wait"] = ...
    random_seed: int | None = ...
    grid_point_recorder: Callable[[tuple[np.int32, ...], np.int32], None] | None = ...
    allow_hbm_allocation_in_run_scoped: bool = ...
    @property
    def num_cores_per_device(self) -> int: ...

@contextlib.contextmanager
def force_tpu_interpret_mode(params: InterpretParams = ...): ...
def set_tpu_interpret_mode(params: InterpretParams = ...): ...

races: RaceDetectionState | None
dma_id_counter: interpret_utils.Counter | None

def reset_tpu_interpret_mode_state() -> None: ...

TPU_MEMORY_SPACE_IDXS: dict[
    mosaic_core.MemorySpace | pallas_core.MemorySpace | None,
    int,
]
TPU_MEMORY_SPACE_NAMES: Incomplete

def get_barrier_semaphore(device_id, collective_id): ...
def get(
    device_id,
    local_core_id,
    memory_space,
    buffer_id,
    transforms,
    block_indices=None,
    grid_loop_idx=None,
    *,
    src_device_id=None,
    src_local_core_id=None,
    clock=None,
    source_info=None,
    input_name=None,
) -> np.ndarray: ...
def store(
    device_id,
    local_core_id,
    memory_space,
    buffer_id,
    transforms,
    val,
    block_indices=None,
    grid_loop_idx=None,
    *,
    src_device_id=None,
    src_local_core_id=None,
    clock=None,
    source_info=None,
    output_name=None,
) -> None: ...
def swap(
    device_id,
    local_core_id,
    memory_space,
    buffer_id,
    transforms,
    val,
    mask,
    *,
    source_info=None,
): ...

class DmaState(enum.Enum):
    STARTED = 0
    READ = 1
    COMPLETED = 2

@dataclasses.dataclass
class DMA:
    id: int
    src_device_id: int
    src_local_core_id: int
    src_memory_space: int
    src_buffer_id: int
    src_transforms: tuple[Any, ...]
    dst_device_id: int
    dst_local_core_id: int
    dst_memory_space: int
    dst_buffer_id: int
    dst_transforms: tuple[Any, ...]
    src_sem: memory.Semaphore | None
    dst_sem: memory.Semaphore
    virtual_device_id: int
    clock: vc.VectorClock
    source_info: source_info_util.SourceInfo | None = ...
    state: DmaState = ...
    data: np.ndarray | None = ...
    lock: threading.Lock = ...
    @property
    def data_size(self) -> int: ...
    @property
    def detect_races(self) -> bool: ...
    @property
    def src_global_core_id(self) -> int: ...
    @property
    def dst_global_core_id(self) -> int: ...
    def execute_read(self) -> None: ...
    def execute_write(self) -> None: ...
    def execute_read_and_write(self) -> None: ...

def dma_start(
    device_id,
    src_local_core_id,
    src_memory_space,
    src_id,
    src_transforms,
    dst_memory_space,
    dst_id,
    dst_transforms,
    dst_sem_id,
    src_sem_id,
    dst_device_id,
    source_info=None,
) -> None: ...
def dma_wait(device_id, local_core_id, sem_id, size) -> None: ...
def semaphore_signal(
    device_id,
    local_core_id,
    sem_id,
    inc,
    target_device_id,
    target_local_core_id,
) -> None: ...
def semaphore_wait(device_id, local_core_id, sem_id, value) -> None: ...

remove_memory_space_p: Incomplete

def get_interpret_effects(): ...
def interpret_pallas_call(
    *args,
    jaxpr: jax_core.Jaxpr,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: pallas_core.GridMapping,
    mesh: pallas_core.Mesh | None,
    compiler_params: pallas_core.CompilerParams | None,
    cost_estimate: pallas_core.CostEstimate,
    out_avals: tuple[jax_core.AbstractValue, ...],
    interpret_params: InterpretParams,
    metadata: frozen_dict.FrozenDict[str, str] | None,
    name: str | None,
): ...
