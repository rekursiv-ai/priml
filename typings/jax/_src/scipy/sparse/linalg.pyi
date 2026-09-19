from jax._src import (
    api as api,
    dtypes as dtypes,
    lax as lax,
)
from jax._src.tree_util import (
    Partial as Partial,
    tree_leaves as tree_leaves,
    tree_map as tree_map,
    tree_reduce as tree_reduce,
    tree_structure as tree_structure,
)
from jax._src.typing import Array as Array

def cg(
    A,
    b,
    x0=None,
    *,
    tol: float = 1e-05,
    atol: float = 0.0,
    maxiter=None,
    M=None,
): ...
def gmres(
    A,
    b,
    x0=None,
    *,
    tol: float = 1e-05,
    atol: float = 0.0,
    restart: int = 20,
    maxiter=None,
    M=None,
    solve_method: str = "batched",
): ...
def bicgstab(
    A,
    b,
    x0=None,
    *,
    tol: float = 1e-05,
    atol: float = 0.0,
    maxiter=None,
    M=None,
): ...
