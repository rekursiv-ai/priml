from jax._src import lax as lax
from jax._src.numpy.util import promote_args_inexact as promote_args_inexact
from jax._src.scipy.special import (
    gammaln as gammaln,
    xlogy as xlogy,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)

def logpmf(k: ArrayLike, n: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array: ...
def pmf(k: ArrayLike, n: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array: ...
