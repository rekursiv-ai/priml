from collections.abc import Sequence

from jax._src import dtypes as dtypes
from jax._src.lib import xla_client as xla_client
from jax._src.numpy import (
    reductions as reductions,
    ufuncs as ufuncs,
)
from jax._src.numpy.util import (
    ensure_arraylike as ensure_arraylike,
    promote_dtypes_inexact as promote_dtypes_inexact,
)
from jax._src.sharding import Sharding as Sharding
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DTypeLike as DTypeLike,
)
from jax._src.util import safe_zip as safe_zip

type Shape = Sequence[int]

def fftn(
    a: ArrayLike,
    s: Shape | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> Array: ...
def ifftn(
    a: ArrayLike,
    s: Shape | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> Array: ...
def rfftn(
    a: ArrayLike,
    s: Shape | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> Array: ...
def irfftn(
    a: ArrayLike,
    s: Shape | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> Array: ...
def fft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Array: ...
def ifft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Array: ...
def rfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Array: ...
def irfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Array: ...
def hfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Array: ...
def ihfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Array: ...
def fft2(
    a: ArrayLike,
    s: Shape | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Array: ...
def ifft2(
    a: ArrayLike,
    s: Shape | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Array: ...
def rfft2(
    a: ArrayLike,
    s: Shape | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Array: ...
def irfft2(
    a: ArrayLike,
    s: Shape | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Array: ...
def fftfreq(
    n: int,
    d: ArrayLike = 1.0,
    *,
    dtype: DTypeLike | None = None,
    device: xla_client.Device | Sharding | None = None,
) -> Array: ...
def rfftfreq(
    n: int,
    d: ArrayLike = 1.0,
    *,
    dtype: DTypeLike | None = None,
    device: xla_client.Device | Sharding | None = None,
) -> Array: ...
def fftshift(x: ArrayLike, axes: int | Sequence[int] | None = None) -> Array: ...
def ifftshift(x: ArrayLike, axes: int | Sequence[int] | None = None) -> Array: ...
