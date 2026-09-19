from collections.abc import Callable as Callable
from typing import Literal, ParamSpec, TypeVar, overload

import contextlib

from _typeshed import Incomplete
from jax._src import (
    stages as stages,
    util as util,
)
from jaxlib.mlir import ir

from .utils import *

T = TypeVar("T")
P = ParamSpec("P")

class Cupti:
    finalize: bool = ...
    def measure(self, f, *, aggregate: bool = True, iterations: int = 1): ...

@overload
def measure(
    f: Callable[P, T],
    *,
    aggregate: Literal[True] = ...,
    iterations: Literal[1] = ...,
) -> Callable[P, tuple[T, float | None]]: ...
@overload
def measure(
    f: Callable[P, T],
    *,
    aggregate: Literal[False] = ...,
    iterations: Literal[1] = ...,
) -> Callable[P, tuple[T, list[tuple[str, float]] | None]]: ...
@overload
def measure(
    f: Callable[P, T],
    *,
    aggregate: Literal[True] = ...,
    iterations: int = ...,
) -> Callable[P, tuple[T, list[float] | None]]: ...
@overload
def measure(
    f: Callable[P, T],
    *,
    aggregate: Literal[False] = ...,
    iterations: int = ...,
) -> Callable[P, tuple[T, list[list[tuple[str, float]]] | None]]: ...

class ProfilerSpec:
    ENTER: int
    EXIT: Incomplete
    entries_per_warpgroup: Incomplete
    interned_names: dict[str, int]
    dump_path: Incomplete
    def __init__(
        self,
        entries_per_warpgroup: int,
        dump_path: str = "sponge",
    ) -> None: ...
    def mlir_buffer_type(
        self,
        grid: tuple[int, ...],
        block: tuple[int, ...],
    ) -> ir.Type: ...
    def jax_buffer_type(
        self,
        grid: tuple[int, ...],
        block: tuple[int, ...],
    ) -> ir.Type: ...
    def smem_i32_elements(self, block: tuple[int, ...]): ...
    def smem_bytes(self, block: tuple[int, ...]): ...
    def intern_name(self, name: str) -> int: ...
    def dump(self, buffer, f, grid: tuple[int, ...], block: tuple[int, ...]): ...

class _ProfilerCtx:
    start: ir.Value
    is_profiling_thread: ir.Value
    smem_buffer: ir.Value
    gmem_buffer: ir.Value
    offset: ir.Value

class OnDeviceProfiler:
    spec: Incomplete
    entries_per_wg: Incomplete
    wrap_in_custom_primitive: Incomplete
    ctx: Incomplete
    def __init__(
        self,
        spec: ProfilerSpec,
        smem_buffer: ir.Value,
        gmem_buffer: ir.Value,
        wrap_in_custom_primitive: bool,
    ) -> None: ...
    @contextlib.contextmanager
    def record(self, name: str): ...
    def finalize(self, grid: tuple[int, ...], block: tuple[int, ...]): ...
