from _typeshed import Incomplete
from jax._src import (
    core as core,
    dtypes as dtypes,
)
from jax._src.lax import lax as lax
from jax._src.numpy import (
    lax_numpy as lax_numpy,
    ufuncs as ufuncs,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)
from jax._src.util import set_module as set_module

export: Incomplete

@export
def blackman(M: int) -> Array: ...
@export
def bartlett(M: int) -> Array: ...
@export
def hamming(M: int) -> Array: ...
@export
def hanning(M: int) -> Array: ...
@export
def kaiser(M: int, beta: ArrayLike) -> Array: ...
