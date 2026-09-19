from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

import abc

from _typeshed import Incomplete
from jax._src import (
    array as array,
    distributed as distributed,
    sharding as sharding,
    typing as typing,
    util as util,
)
from jax._src.layout import Format as Format
from jax.experimental.array_serialization import tensorstore_impl as ts_impl
from jax.experimental.array_serialization.tensorstore_impl import (
    async_deserialize as async_deserialize,
    async_serialize as async_serialize,
)

import jax

get_tensorstore_spec: Incomplete

class BarrierTimeoutError(Exception): ...

logger: Incomplete

def is_remote_storage(tspec: dict[str, Any] | str) -> bool: ...

class GlobalAsyncCheckpointManagerBase(util.StrictABC, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def check_for_errors(self): ...
    @abc.abstractmethod
    def wait_until_finished(self): ...
    @abc.abstractmethod
    def serialize(
        self,
        arrays,
        tensorstore_specs,
        *,
        on_commit_callback: Callable[[], None],
    ): ...
    @abc.abstractmethod
    def deserialize(
        self,
        shardings: Sequence[sharding.Sharding],
        tensorstore_specs: Sequence[dict[str, Any]],
        global_shapes: Sequence[array.Shape] | None = None,
        dtypes: Sequence[typing.DTypeLike] | None = None,
    ): ...

class AsyncManager:
    def __init__(self, timeout_secs: int = 300) -> None: ...
    def __del__(self) -> None: ...
    def check_for_errors(self) -> None: ...
    def wait_until_finished(self) -> None: ...

class GlobalAsyncCheckpointManager(AsyncManager, GlobalAsyncCheckpointManagerBase):
    def serialize(
        self,
        arrays,
        tensorstore_specs,
        *,
        on_commit_callback: Callable[[], None] | None = None,
        transaction: ts_impl.Transaction | None = None,
    ): ...
    def serialize_with_paths(
        self,
        arrays: Sequence[jax.Array],
        paths: Sequence[str],
        *,
        on_commit_callback: Callable[[], None] | None = None,
        transaction: ts_impl.Transaction | None = None,
    ): ...
    def deserialize(
        self,
        shardings: Sequence[sharding.Sharding | Format],
        tensorstore_specs: Sequence[dict[str, Any]],
        global_shapes: Sequence[array.Shape] | None = None,
        dtypes: Sequence[typing.DTypeLike] | None = None,
        concurrent_gb: int = 32,
    ): ...
    def deserialize_with_paths(
        self,
        shardings: Sequence[sharding.Sharding],
        paths: Sequence[str],
        global_shapes: Sequence[array.Shape] | None = None,
        dtypes: Sequence[typing.DTypeLike] | None = None,
        concurrent_gb: int = 32,
    ): ...
