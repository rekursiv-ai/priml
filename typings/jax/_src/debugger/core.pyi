from collections.abc import Hashable
from typing import Any, Protocol

import dataclasses

from _typeshed import Incomplete
from jax._src import (
    callback as callback,
    core as core,
    debugging as debugging,
    traceback_util as traceback_util,
    tree_util as tree_util,
    util as util,
)
from jax._src.lax import lax as lax

class _DictWrapper:
    keys: list[Hashable]
    values: list[Any]
    def __init__(self, keys, values) -> None: ...
    def to_dict(self): ...
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, keys, values): ...

class _CantFlatten: ...

cant_flatten: Incomplete

@dataclasses.dataclass(frozen=True)
class DebuggerFrame:
    filename: str
    locals: dict[str, Any]
    globals: dict[str, Any]
    code_context: str
    source: list[str]
    lineno: int
    offset: int | None
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, info, valid_vars): ...
    @classmethod
    def from_frameinfo(cls, frame_info) -> DebuggerFrame: ...

class Debugger(Protocol):
    def __call__(
        self,
        frames: list[DebuggerFrame],
        thread_id: int | None,
        **kwargs: Any,
    ) -> None: ...

def get_debugger(backend: str | None = None) -> Debugger: ...
def register_debugger(name: str, debugger: Debugger, priority: int) -> None: ...

debug_lock: Incomplete

def breakpoint(
    *,
    backend: str | None = None,
    filter_frames: bool = True,
    num_frames: int | None = None,
    ordered: bool = False,
    token=None,
    **kwargs,
): ...
