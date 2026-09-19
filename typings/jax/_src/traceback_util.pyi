from collections.abc import Callable
from typing import Any, TypeVar

import types

from _typeshed import Incomplete
from jax._src import (
    config as config,
    repro as repro,
    util as util,
)

C = TypeVar("C", bound=Callable[..., Any])

def register_exclusion(path: str): ...
def include_frame(f: types.FrameType) -> bool: ...
def include_filename(filename: str) -> bool: ...
def filter_traceback(tb: types.TracebackType) -> types.TracebackType | None: ...
def format_exception_only(e: BaseException) -> str: ...

class UnfilteredStackTrace(Exception): ...
class SimplifiedTraceback(Exception): ...

def api_boundary(
    fun: C,
    *,
    repro_api_name: str | None = None,
    repro_user_func: bool = False,
) -> C: ...

repro_is_enabled: Incomplete
