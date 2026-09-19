from collections.abc import (
    Callable as Callable,
    Sequence,
)

from jax._src import (
    api as api,
    core as core,
    dtypes as dtypes,
    lax as lax,
)
from jax._src.lax.lax import PrecisionLike as PrecisionLike
from jax._src.numpy import linalg as linalg
from jax._src.numpy.util import (
    check_arraylike as check_arraylike,
    ensure_arraylike as ensure_arraylike,
    promote_dtypes_complex as promote_dtypes_complex,
    promote_dtypes_inexact as promote_dtypes_inexact,
)
from jax._src.third_party.scipy import signal_helper as signal_helper
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)
from jax._src.util import (
    canonicalize_axis as canonicalize_axis,
    tuple_delete as tuple_delete,
    tuple_insert as tuple_insert,
)

def fftconvolve(
    in1: ArrayLike,
    in2: ArrayLike,
    mode: str = "full",
    axes: Sequence[int] | None = None,
) -> Array: ...
def convolve(
    in1: Array,
    in2: Array,
    mode: str = "full",
    method: str = "auto",
    precision: PrecisionLike = None,
) -> Array: ...
def convolve2d(
    in1: Array,
    in2: Array,
    mode: str = "full",
    boundary: str = "fill",
    fillvalue: float = 0,
    precision: PrecisionLike = None,
) -> Array: ...
def correlate(
    in1: Array,
    in2: Array,
    mode: str = "full",
    method: str = "auto",
    precision: PrecisionLike = None,
) -> Array: ...
def correlate2d(
    in1: Array,
    in2: Array,
    mode: str = "full",
    boundary: str = "fill",
    fillvalue: float = 0,
    precision: PrecisionLike = None,
) -> Array: ...
def detrend(
    data: ArrayLike,
    axis: int = -1,
    type: str = "linear",
    bp: int = 0,
    overwrite_data: None = None,
) -> Array: ...
def odd_ext(x: Array, n: int, axis: int = -1) -> Array: ...
def stft(
    x: Array,
    fs: ArrayLike = 1.0,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: bool = False,
    return_onesided: bool = True,
    boundary: str | None = "zeros",
    padded: bool = True,
    axis: int = -1,
) -> tuple[Array, Array, Array]: ...
def csd(
    x: Array,
    y: ArrayLike | None,
    fs: ArrayLike = 1.0,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str = "constant",
    return_onesided: bool = True,
    scaling: str = "density",
    axis: int = -1,
    average: str = "mean",
) -> tuple[Array, Array]: ...
def welch(
    x: Array,
    fs: ArrayLike = 1.0,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str = "constant",
    return_onesided: bool = True,
    scaling: str = "density",
    axis: int = -1,
    average: str = "mean",
) -> tuple[Array, Array]: ...
def istft(
    Zxx: Array,
    fs: ArrayLike = 1.0,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    input_onesided: bool = True,
    boundary: bool = True,
    time_axis: int = -1,
    freq_axis: int = -2,
) -> tuple[Array, Array]: ...
