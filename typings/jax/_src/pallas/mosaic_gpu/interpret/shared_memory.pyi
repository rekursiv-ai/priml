from typing import Any

import dataclasses

from _typeshed import Incomplete
from jax._src.pallas.mosaic.interpret import (
    shared_memory as memory,
    utils as interpret_utils,
    vector_clock as vc,
)

class Barrier(memory.Allocation):
    shared_memory: Incomplete
    ref_count: int
    num_arrivals: int
    arrivals_count: int
    enable_logging: bool
    phase: int
    next_awaited_phase_by_thread: list[int]
    phase_change_observed: bool
    cv: Incomplete
    clock: vc.VectorClock | None
    def __init__(
        self,
        shared_memory: GPUSharedMemory,
        ref_count: int,
        num_arrivals: int,
        enable_logging: bool = False,
    ) -> None: ...
    @property
    def detect_races(self) -> bool: ...
    def has_zero_ref_count(self) -> bool: ...
    def deallocate(self) -> None: ...
    def arrive(self, device_id: int, local_thread_id: int, clock): ...
    def wait(self, device_id: int, local_thread_id: int): ...

@dataclasses.dataclass
class GPUSharedMemory(memory.SharedMemory):
    logging_mode: interpret_utils.LoggingMode | None = ...
    @property
    def num_threads_per_device(self) -> int: ...
    @property
    def num_global_threads(self) -> int: ...
    def get_global_thread_id(self, device_id: int, local_thread_id: int) -> int: ...
    def allocate_barrier(
        self,
        device_id: int,
        thread_id: int,
        key: Any,
        ref_count: int,
        num_arrivals: int,
    ): ...
    def get_barrier_and_increment_clock(
        self,
        key: Any,
        device_id: int,
        thread_id: int,
    ) -> tuple[Barrier, vc.VectorClock | None]: ...
    def deallocate_barrier(self, device_id: int, thread_id: int, key: Any): ...
    def assert_no_barriers_allocated(self) -> None: ...
