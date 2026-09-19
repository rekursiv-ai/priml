from jax._src import lax as lax
from jax._src.numpy.util import (
    ensure_arraylike as ensure_arraylike,
    promote_args_inexact as promote_args_inexact,
    promote_dtypes_inexact as promote_dtypes_inexact,
)
from jax._src.scipy.special import (
    entr as entr,
    gammaincc as gammaincc,
    gammaln as gammaln,
    xlogy as xlogy,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)

def logpmf(k: ArrayLike, mu: ArrayLike, loc: ArrayLike = 0) -> Array: ...
def pmf(k: ArrayLike, mu: ArrayLike, loc: ArrayLike = 0) -> Array: ...
def cdf(k: ArrayLike, mu: ArrayLike, loc: ArrayLike = 0) -> Array: ...
def entropy(mu: ArrayLike, loc: ArrayLike = 0) -> Array: ...
