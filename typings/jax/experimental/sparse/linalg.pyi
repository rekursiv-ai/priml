from collections.abc import Callable as Callable

from _typeshed import Incomplete
from jax._src import (
    core as core,
    dispatch as dispatch,
    ffi as ffi,
)
from jax._src.interpreters import ad as ad
from jax.experimental import sparse as sparse
from jax.interpreters import mlir as mlir

import jax

def lobpcg_standard(
    A: jax.Array | Callable[[jax.Array], jax.Array],
    X: jax.Array,
    m: int = 100,
    tol: jax.Array | float | None = None,
): ...

spsolve_p: Incomplete

def spsolve(data, indices, indptr, b, tol: float = 1e-06, reorder: int = 1): ...
