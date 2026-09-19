from jax._src import lax as lax
from jax._src.numpy.util import promote_args_inexact as promote_args_inexact
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)

def logpdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array: ...
def pdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array: ...
def cdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array: ...
