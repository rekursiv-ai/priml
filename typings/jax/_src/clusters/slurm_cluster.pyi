from jax._src import clusters as clusters

class SlurmCluster(clusters.ClusterEnv):
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
    def get_process_count(cls) -> int: ...
    @classmethod
    def get_process_id(cls) -> int: ...
    @classmethod
    def get_local_process_id(cls) -> int | None: ...
