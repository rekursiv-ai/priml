from collections.abc import Sequence
from typing import TypeAlias

from _typeshed import Incomplete
from jax._src import state as state
from jax._src.pallas.triton import lowering as lowering
from jax.interpreters import mlir as mlir

import jax

Ref: TypeAlias

def approx_tanh(x: jax.Array) -> jax.Array: ...
def elementwise_inline_asm(
    asm: str,
    *,
    args: Sequence[jax.Array],
    constraints: str,
    pack: int,
    result_shape_dtypes: Sequence[jax.ShapeDtypeStruct],
) -> Sequence[jax.Array]: ...

elementwise_inline_asm_p: Incomplete

def debug_barrier() -> None: ...

debug_barrier_p: Incomplete

def load(
    ref: Ref,
    *,
    mask: jax.Array | None = None,
    other: jax.typing.ArrayLike | None = None,
    cache_modifier: str | None = None,
    eviction_policy: str | None = None,
    volatile: bool = False,
) -> jax.Array: ...
def store(
    ref: Ref,
    val: jax.Array,
    *,
    mask: jax.Array | None = None,
    eviction_policy: str | None = None,
) -> None: ...
