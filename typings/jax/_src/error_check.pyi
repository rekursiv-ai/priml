import dataclasses
import threading
import types

from _typeshed import Incomplete
from jax._src import (
    core as core,
    shard_map as shard_map,
    source_info_util as source_info_util,
    traceback_util as traceback_util,
    tree_util as tree_util,
)
from jax._src.lax import lax as lax
from jax._src.sharding_impls import NamedSharding as NamedSharding
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)

class JaxValueError(ValueError): ...

class _ErrorStorage(threading.local):
    ref: core.Ref | None
    def __init__(self) -> None: ...

class error_checking_context:
    old_ref: Incomplete
    def __init__(self) -> None: ...
    def __enter__(self): ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...

def set_error_if(pred: Array, /, msg: str) -> None: ...
def raise_if_error() -> None: ...

@dataclasses.dataclass(frozen=True)
class _ErrorClass:
    error_code: Array
    error_list: list[tuple[str, str]]

def wrap_for_export(f): ...
def unwrap_from_import(f): ...
