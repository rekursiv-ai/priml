from enum import Enum
from typing import Any

class ONNXRuntimeError(Exception): ...

class GraphOptimizationLevel(Enum):
    ORT_DISABLE_ALL: GraphOptimizationLevel

class SessionOptions:
    enable_profiling: bool
    graph_optimization_level: GraphOptimizationLevel
    profile_file_prefix: str

    def __init__(self) -> None: ...

class InferenceSession:
    def __init__(
        self,
        path_or_bytes: str | bytes,
        sess_options: SessionOptions | None = None,
        **kwargs: Any,
    ) -> None: ...
    def run(self, output_names: list[str], input_feed: dict[str, Any]) -> list[Any]: ...
    def end_profiling(self) -> str: ...
