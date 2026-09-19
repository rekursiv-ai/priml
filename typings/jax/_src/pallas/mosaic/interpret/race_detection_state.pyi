import dataclasses
import threading

from jax._src import source_info_util as source_info_util

@dataclasses.dataclass
class RaceDetectionState:
    num_cores: int
    reads: dict = ...
    writes: dict = ...
    lock: threading.Lock = ...
    races_found: bool = ...
    def check_read(
        self,
        device_id,
        local_core_id,
        clock,
        buffer_key,
        rnge,
        source_info=None,
    ) -> None: ...
    def check_write(
        self,
        device_id,
        local_core_id,
        clock,
        buffer_key,
        rnge,
        source_info=None,
    ) -> None: ...
