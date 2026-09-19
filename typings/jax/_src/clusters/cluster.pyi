from collections.abc import Sequence

from _typeshed import Incomplete
from jax._src.cloud_tpu_init import running_in_cloud_tpu_vm as running_in_cloud_tpu_vm

logger: Incomplete

class ClusterEnv:
    name: str
    opt_in_only_method: bool
    def __init_subclass__(cls, **kwargs) -> None: ...
    @classmethod
    def auto_detect_unset_distributed_params(
        cls,
        coordinator_address: str | None,
        num_processes: int | None,
        process_id: int | None,
        local_device_ids: Sequence[int] | None,
        cluster_detection_method: str | None,
        initialization_timeout: int | None,
    ) -> tuple[str | None, int | None, int | None, Sequence[int] | None]: ...
    @classmethod
    def is_env_present(cls) -> bool: ...
    @classmethod
    def get_coordinator_address(
        cls,
        timeout_secs: int | None,
        override_coordinator_port: str | None,
    ) -> str: ...
    @classmethod
    def get_process_count(cls) -> int: ...
    @classmethod
    def get_process_id(cls) -> int: ...
    @classmethod
    def get_local_process_id(cls) -> int | None: ...
