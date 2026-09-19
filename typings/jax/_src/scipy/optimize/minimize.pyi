from collections.abc import (
    Callable as Callable,
    Mapping,
)
from typing import Any, NamedTuple

from jax._src.scipy.optimize.bfgs import minimize_bfgs as minimize_bfgs
from jax._src.typing import Array as Array

class OptimizeResults(NamedTuple):
    x: Array
    success: bool | Array
    status: int | Array
    fun: Array
    jac: Array
    hess_inv: Array | None
    nfev: int | Array
    njev: int | Array
    nit: int | Array

def minimize(
    fun: Callable,
    x0: Array,
    args: tuple = (),
    *,
    method: str,
    tol: float | None = None,
    options: Mapping[str, Any] | None = None,
) -> OptimizeResults: ...
