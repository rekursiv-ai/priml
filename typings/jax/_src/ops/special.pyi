from typing import Literal, overload

from jax._src import config as config
from jax._src.lax import lax as lax
from jax._src.numpy import (
    reductions as reductions,
    ufuncs as ufuncs,
)
from jax._src.numpy.reductions import Axis as Axis
from jax._src.numpy.util import promote_args_inexact as promote_args_inexact
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)

@overload
def logsumexp(
    a: ArrayLike,
    axis: Axis = None,
    b: ArrayLike | None = None,
    keepdims: bool = False,
    return_sign: Literal[False] = False,
    where: ArrayLike | None = None,
) -> Array: ...
@overload
def logsumexp(
    a: ArrayLike,
    axis: Axis = None,
    b: ArrayLike | None = None,
    keepdims: bool = False,
    *,
    return_sign: Literal[True],
    where: ArrayLike | None = None,
) -> tuple[Array, Array]: ...
@overload
def logsumexp(
    a: ArrayLike,
    axis: Axis = None,
    b: ArrayLike | None = None,
    keepdims: bool = False,
    return_sign: bool = False,
    where: ArrayLike | None = None,
) -> Array | tuple[Array, Array]: ...
