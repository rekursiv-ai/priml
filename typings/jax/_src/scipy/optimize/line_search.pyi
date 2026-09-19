from collections.abc import Callable
from typing import NamedTuple

from jax._src import (
    api as api,
    dtypes as dtypes,
    lax as lax,
)
from jax._src.numpy.util import promote_dtypes_inexact as promote_dtypes_inexact
from jax._src.typing import Array as Array

class _ZoomState(NamedTuple):
    done: bool | Array
    failed: bool | Array
    j: int | Array
    a_lo: float | Array
    phi_lo: float | Array
    dphi_lo: float | Array
    a_hi: float | Array
    phi_hi: float | Array
    dphi_hi: float | Array
    a_rec: float | Array
    phi_rec: float | Array
    a_star: float | Array
    phi_star: float | Array
    dphi_star: float | Array
    g_star: float | Array
    nfev: int | Array
    ngev: int | Array

type ConditionFn = Callable[..., Array]

class _LineSearchState(NamedTuple):
    done: Array
    failed: Array
    i: int | Array
    a_i1: float | Array
    phi_i1: float | Array
    dphi_i1: float | Array
    nfev: int | Array
    ngev: int | Array
    a_star: float | Array
    phi_star: Array
    dphi_star: Array
    g_star: Array

class _LineSearchResults(NamedTuple):
    failed: bool | Array
    nit: int | Array
    nfev: int | Array
    ngev: int | Array
    k: int | Array
    a_k: int | Array
    f_k: Array
    g_k: Array
    status: bool | Array

def line_search(
    f,
    xk,
    pk,
    old_fval=None,
    old_old_fval=None,
    gfk=None,
    c1: float = 0.0001,
    c2: float = 0.9,
    maxiter: int = 20,
): ...
