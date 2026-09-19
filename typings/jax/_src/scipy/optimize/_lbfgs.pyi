from collections.abc import Callable as Callable
from typing import NamedTuple

from jax._src import (
    api as api,
    dtypes as dtypes,
    lax as lax,
)
from jax._src.scipy.optimize.line_search import line_search as line_search
from jax._src.typing import Array as Array

class LBFGSResults(NamedTuple):
    converged: Array
    failed: Array
    k: int | Array
    nfev: int | Array
    ngev: int | Array
    x_k: Array
    f_k: Array
    g_k: Array
    s_history: Array
    y_history: Array
    rho_history: Array
    gamma: float | Array
    status: int | Array
    ls_status: int | Array
