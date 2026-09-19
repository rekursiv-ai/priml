from collections.abc import Callable as Callable

from jax import (
    api_util as api_util,
    custom_derivatives as custom_derivatives,
    lax as lax,
)
from jax._src import (
    core as core,
    linear_util as lu,
)
from jax._src.numpy.util import promote_dtypes_inexact as promote_dtypes_inexact
from jax._src.util import (
    safe_map as safe_map,
    safe_zip as safe_zip,
)
from jax.flatten_util import ravel_pytree as ravel_pytree
from jax.tree_util import (
    tree_leaves as tree_leaves,
    tree_map as tree_map,
)

map = safe_map
zip = safe_zip

def ravel_first_arg(f: Callable, unravel, debug_info: core.DebugInfo): ...
@lu.transformation2
def ravel_first_arg_(f, unravel, y_flat, *args): ...
def interp_fit_dopri(y0, y1, k, dt): ...
def fit_4th_order_polynomial(y0, y1, y_mid, dy0, dy1, dt): ...
def initial_step_size(fun, t0, y0, order, rtol, atol, f0): ...
def runge_kutta_step(func, y0, f0, t0, dt): ...
def abs2(x): ...
def mean_error_ratio(error_estimate, rtol, atol, y0, y1): ...
def optimal_step_size(
    last_step,
    mean_error_ratio,
    safety: float = 0.9,
    ifactor: float = 10.0,
    dfactor: float = 0.2,
    order: float = 5.0,
): ...
def odeint(
    func,
    y0,
    t,
    *args,
    rtol: float = 1.4e-08,
    atol: float = 1.4e-08,
    mxstep=...,
    hmax=...,
): ...
