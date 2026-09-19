from os import PathLike
from typing import Self

class TraceAnnotation:
    def __init__(self, name: str, **metadata: object) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *exc_info: object) -> None: ...

class ProfileOptions:
    host_tracer_level: int
    device_tracer_level: int
    python_tracer_level: int

def start_trace(
    log_dir: PathLike[str] | str,
    create_perfetto_link: bool = False,
    create_perfetto_trace: bool = False,
    profiler_options: ProfileOptions | None = None,
) -> None: ...
def stop_trace() -> None: ...
def save_device_memory_profile(
    filename: PathLike[str] | str,
    backend: str | None = None,
) -> None: ...
