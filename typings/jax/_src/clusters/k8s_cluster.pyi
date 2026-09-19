from _typeshed import Incomplete
from jax._src import clusters as clusters

logger: Incomplete

def retry(func=None, initial_delay: int = 0, wait=..., exceptions=...): ...

class K8sCluster(clusters.ClusterEnv):
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
