from jax._src import lax as lax
from jax._src.numpy.util import promote_args_inexact as promote_args_inexact
from jax._src.scipy.special import (
    gammainc as gammainc,
    gammaincc as gammaincc,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)

def logpdf(
    x: ArrayLike,
    df: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array: ...
def pdf(
    x: ArrayLike,
    df: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array: ...
def cdf(
    x: ArrayLike,
    df: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array: ...
def logcdf(
    x: ArrayLike,
    df: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array: ...
def sf(
    x: ArrayLike,
    df: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array: ...
def logsf(
    x: ArrayLike,
    df: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array: ...
