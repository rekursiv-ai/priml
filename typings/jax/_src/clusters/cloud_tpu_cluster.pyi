from _typeshed import Incomplete
from jax._src import clusters as clusters
from jax._src.cloud_tpu_init import running_in_cloud_tpu_vm as running_in_cloud_tpu_vm

logger: Incomplete
coordinator_port: str
metadata_response_code_success: int

def get_metadata(key): ...
def get_tpu_env_value_from_metadata(key) -> str | None: ...
def get_tpu_env_value(key) -> str | None: ...

class BaseTpuCluster(clusters.ClusterEnv):
    name: str
    @classmethod
    def is_env_present(cls) -> bool: ...
    @classmethod
    def get_coordinator_address(
        cls,
        timeout_secs: int | None,
        override_coordinator_port: str | None,
    ) -> str: ...
    @classmethod
    def wait_for_coordinator(cls, coordinator_address, timeout_secs) -> None: ...
    @classmethod
    def get_process_count(cls) -> int: ...
    @classmethod
    def get_process_id(cls) -> int: ...

class GceTpuCluster(BaseTpuCluster):
    name: str
    @classmethod
    def is_env_present(cls) -> bool: ...

class GkeTpuCluster(BaseTpuCluster):
    name: str
    @classmethod
    def is_env_present(cls) -> bool: ...
