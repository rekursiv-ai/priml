from collections.abc import (
    Callable as Callable,
    Hashable,
    Sequence,
)
from typing import overload

import dataclasses

from jax import lax as lax
from jax._src import dtypes as dtypes

import jax

@dataclasses.dataclass(frozen=True, eq=False)
class NDLoopInfo:
    index: tuple[jax.Array, ...]
    local_index: jax.Array | int
    num_local_steps: jax.Array | int | None

@overload
def nd_loop(
    grid: Sequence[int],
    *,
    collective_axes: Sequence[Hashable] | Hashable,
    tiling: Sequence[int] | None = None,
    init_carry: None = None,
) -> Callable[[Callable[[NDLoopInfo], None]], None]: ...
@overload
def nd_loop(
    grid: Sequence[int],
    *,
    collective_axes: Sequence[Hashable] | Hashable,
    tiling: Sequence[int] | None = None,
    init_carry: _T,
) -> Callable[[Callable[[NDLoopInfo, _T], _T]], _T]: ...
def format_tcgen05_sparse_metadata(meta): ...
def find_swizzle(minor_dim_bits: int, what: str = ""): ...
def planar_snake(
    lin_idx: jax.Array,
    shape: tuple[int, int],
    minor_dim: int,
    tile_width: int,
): ...
@overload
def dynamic_scheduling_loop(
    grid_names: Sequence[Hashable],
    *,
    thread_axis: Hashable | None = None,
    init_carry: None = None,
) -> Callable[[Callable[[NDLoopInfo], None]], None]: ...
@overload
def dynamic_scheduling_loop(
    grid_names: Sequence[Hashable],
    *,
    thread_axis: Hashable | None = None,
    init_carry: _T,
) -> Callable[[Callable[[NDLoopInfo, _T], _T]], _T]: ...
