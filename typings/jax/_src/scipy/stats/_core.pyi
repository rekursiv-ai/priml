from typing import NamedTuple

from jax._src import (
    api as api,
    dtypes as dtypes,
    lax as lax,
)
from jax._src.numpy.util import (
    check_arraylike as check_arraylike,
    promote_args_inexact as promote_args_inexact,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)
from jax._src.util import canonicalize_axis as canonicalize_axis

class ModeResult(NamedTuple):
    mode: Array
    count: Array

def mode(
    a: ArrayLike,
    axis: int | None = 0,
    nan_policy: str = "propagate",
    keepdims: bool = False,
) -> ModeResult: ...
def invert_permutation(i: Array) -> Array: ...
def rankdata(
    a: ArrayLike,
    method: str = "average",
    *,
    axis: int | None = None,
    nan_policy: str = "propagate",
) -> Array: ...
def sem(
    a: ArrayLike,
    axis: int | None = 0,
    ddof: int = 1,
    nan_policy: str = "propagate",
    *,
    keepdims: bool = False,
) -> Array: ...
