from collections.abc import Sequence

from jax._src import lax as lax
from jax._src.numpy.util import (
    ensure_arraylike as ensure_arraylike,
    promote_dtypes_complex as promote_dtypes_complex,
    promote_dtypes_inexact as promote_dtypes_inexact,
)
from jax._src.typing import Array as Array
from jax._src.util import (
    canonicalize_axis as canonicalize_axis,
    canonicalize_axis_tuple as canonicalize_axis_tuple,
)

def dct(
    x: Array,
    type: int = 2,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Array: ...
def dctn(
    x: Array,
    type: int = 2,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> Array: ...
def idct(
    x: Array,
    type: int = 2,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Array: ...
def idctn(
    x: Array,
    type: int = 2,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> Array: ...
