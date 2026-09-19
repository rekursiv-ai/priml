from typing import Any

from _typeshed import Incomplete
from jax._src.compilation_cache_interface import CacheInterface as CacheInterface

filelock: Any | None
logger: Incomplete

class LRUCache(CacheInterface):
    path: Incomplete
    eviction_enabled: Incomplete
    max_size: Incomplete
    lock_timeout_secs: Incomplete
    lock_path: Incomplete
    lock: Incomplete
    def __init__(
        self,
        path: str,
        *,
        max_size: int,
        lock_timeout_secs: float | None = 10,
    ) -> None: ...
    def get(self, key: str) -> bytes | None: ...
    def put(self, key: str, val: bytes) -> None: ...
