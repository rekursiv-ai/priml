from jax._src import lax as lax
from jax._src.numpy.util import promote_args_inexact as promote_args_inexact
from jax._src.scipy.special import (
    betainc as betainc,
    betaln as betaln,
    xlog1py as xlog1py,
    xlogy as xlogy,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)

def logpdf(
    x: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array: ...
def pdf(
    x: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array: ...
def cdf(
    x: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array: ...
def logcdf(
    x: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array: ...
def sf(
    x: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array: ...
def logsf(
    x: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array: ...
