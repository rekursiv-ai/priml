from jax._src import lax as lax
from jax._src.numpy.util import promote_dtypes_inexact as promote_dtypes_inexact
from jax._src.scipy.special import (
    gammaln as gammaln,
    xlogy as xlogy,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)

def logpdf(x: ArrayLike, alpha: ArrayLike) -> Array: ...
def pdf(x: ArrayLike, alpha: ArrayLike) -> Array: ...
