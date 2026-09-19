from collections.abc import Sequence
from typing import Any

from _typeshed import Incomplete
from jax._src import (
    clusters as clusters,
    config as config,
    xla_bridge as xla_bridge,
)
from jax._src.lib import (
    _jax,
    jaxlib_extension_version as jaxlib_extension_version,
)

logger: Incomplete

class State:
    process_id: int
    num_processes: int
    service: _jax.DistributedRuntimeService | Any | None
    client: _jax.DistributedRuntimeClient | Any | None
    preemption_sync_manager: Any | None
    coordinator_address: str | None
    partition_index: int | None
    def initialize(
        self,
        coordinator_address: str | None = None,
        num_processes: int | None = None,
        process_id: int | None = None,
        local_device_ids: int | Sequence[int] | None = None,
        cluster_detection_method: str | None = None,
        initialization_timeout: int = 300,
        coordinator_bind_address: str | None = None,
        heartbeat_timeout_seconds: int = 100,
        shutdown_timeout_seconds: int = 300,
        partition_index: int | None = None,
    ): ...
    def shutdown(self) -> None: ...
    def initialize_preemption_sync_manager(self) -> None: ...

global_state: Incomplete

def initialize(
    coordinator_address: str | None = None,
    num_processes: int | None = None,
    process_id: int | None = None,
    local_device_ids: int | Sequence[int] | None = None,
    cluster_detection_method: str | None = None,
    initialization_timeout: int = 300,
    heartbeat_timeout_seconds: int = 100,
    shutdown_timeout_seconds: int = 300,
    coordinator_bind_address: str | None = None,
    slice_index: int | None = None,
    partition_index: int | None = None,
): ...
def is_initialized() -> bool: ...
def shutdown() -> None: ...
