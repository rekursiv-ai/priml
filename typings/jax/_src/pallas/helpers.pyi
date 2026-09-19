from collections.abc import (
    Callable as Callable,
    Hashable,
)
from typing import Any

from _typeshed import Incomplete
from jax._src import (
    api as api,
    checkify as checkify,
    config as config,
    tree_util as tree_util,
    typing as jax_typing,
    util as util,
)
from jax._src.lax.control_flow import conditionals as conditionals
from jax._src.pallas import core as pl_core

empty: Incomplete

@api.named_call
def empty_like(x: object): ...
def empty_ref_like(x: object) -> jax_typing.Array: ...
def when(
    condition: bool | jax_typing.ArrayLike,
    /,
) -> Callable[[Callable[[], None]], Callable[[], None]]: ...
def loop(
    lower: jax_typing.ArrayLike,
    upper: jax_typing.ArrayLike,
    *,
    step: jax_typing.ArrayLike = 1,
    unroll: int | bool | None = None,
) -> Callable[[Callable[[jax_typing.Array], None]], None]: ...

enable_debug_checks: Incomplete

def debug_checks_enabled() -> bool: ...
def debug_check(condition, message): ...
def kernel(
    body: Callable | api.NotSpecified = ...,
    out_shape: object | None = None,
    *,
    mesh: pl_core.Mesh,
    scratch_shapes: pl_core.ScratchShapeTree = (),
    compiler_params: pl_core.CompilerParams | None = None,
    interpret: bool = False,
    cost_estimate: pl_core.CostEstimate | None = None,
    debug: bool = False,
    name: str | None = None,
    metadata: dict[str, str] | None = None,
): ...
def with_scoped(
    *types: Any,
    collective_axes: Hashable | tuple[Hashable, ...] = (),
    **kw_types: Any,
): ...
