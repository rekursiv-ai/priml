from collections.abc import Callable
from typing import Any, Protocol

from _typeshed import Incomplete
from jax import (
    lax as lax,
    nn as nn,
    numpy as numpy,
    random as random,
    tree as tree,
    tree_util as tree_util,
)
from jax._src.api import (
    eval_shape as eval_shape,
    named_scope as named_scope,
)
from jax._src.stages import Lowered, Traced
from jax.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
    partial_eval as partial_eval,
    pxla as pxla,
    xla as xla,
)

Array: Incomplete
Device: Incomplete
device_put_replicated: Incomplete
device_put_sharded: Incomplete

class _Config(Protocol):
    jax_compilation_cache_dir: str | None
    jax_compilation_cache_include_metadata_in_key: bool
    def update(self, name: str, value: object) -> None: ...

config: _Config

def device_get(x: object) -> object: ...
def devices(backend: str | None = ...) -> list[Any]: ...
def local_devices(process_index: int = ..., backend: str | None = ...) -> list[Any]: ...
def device_count(backend: str | None = ...) -> int: ...
def default_backend() -> str: ...
def clear_caches() -> None: ...
def effects_barrier() -> None: ...
def block_until_ready(x: Any) -> Any: ...

class Jitted[**P, R](Protocol):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    def trace(self, *args: P.args, **kwargs: P.kwargs) -> Traced: ...
    def lower(self, *args: P.args, **kwargs: P.kwargs) -> Lowered: ...

def jit[**P, R](__fun: Callable[P, R], *args: Any, **kwargs: Any) -> Jitted[P, R]: ...
def pmap[**P, R](
    __fun: Callable[P, R],
    axis_name: str | None = ...,
    *args: Any,
    **kwargs: Any,
) -> Jitted[P, R]: ...
def vmap(
    __fun: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Callable[..., Any]: ...
def grad(
    __fun: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Callable[..., Any]: ...
def value_and_grad(
    __fun: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Callable[..., Any]: ...
def checkpoint(
    __fun: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Callable[..., Any]: ...
