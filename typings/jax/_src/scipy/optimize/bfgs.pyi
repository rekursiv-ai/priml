from collections.abc import Callable as Callable
from typing import NamedTuple

from jax._src import (
    api as api,
    lax as lax,
)
from jax._src.scipy.optimize.line_search import line_search as line_search
from jax._src.typing import Array as Array

class _BFGSResults(NamedTuple):
    converged: bool | Array
    failed: bool | Array
    k: int | Array
    nfev: int | Array
    ngev: int | Array
    nhev: int | Array
    x_k: Array
    f_k: Array
    g_k: Array
    H_k: Array
    old_old_fval: Array
    status: int | Array
    line_search_status: int | Array

def minimize_bfgs(
    fun: Callable,
    x0: Array,
    maxiter: int | None = None,
    norm=...,
    gtol: float = 1e-05,
    line_search_maxiter: int = 10,
) -> _BFGSResults: ...
