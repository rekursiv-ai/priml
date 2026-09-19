from jax._src import lax as lax
from jax._src.numpy.util import promote_dtypes_inexact as promote_dtypes_inexact
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)

def logpdf(
    x: ArrayLike,
    mean: ArrayLike,
    cov: ArrayLike,
    allow_singular: None = None,
) -> ArrayLike: ...
def pdf(x: ArrayLike, mean: ArrayLike, cov: ArrayLike) -> Array: ...
