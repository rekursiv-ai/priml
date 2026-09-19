from jax._src import api as api
from jax._src.numpy.util import (
    check_arraylike as check_arraylike,
    promote_dtypes_inexact as promote_dtypes_inexact,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)

def vq(
    obs: ArrayLike,
    code_book: ArrayLike,
    check_finite: bool = True,
) -> tuple[Array, Array]: ...
