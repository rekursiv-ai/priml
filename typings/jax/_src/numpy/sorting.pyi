from collections.abc import Sequence

from _typeshed import Incomplete
from jax._src import (
    api as api,
    dtypes as dtypes,
)
from jax._src.lax import lax as lax
from jax._src.numpy import util as util
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DTypeLike as DTypeLike,
)
from jax._src.util import (
    canonicalize_axis as canonicalize_axis,
    set_module as set_module,
)

import numpy as np

export: Incomplete

@export
def sort(
    a: ArrayLike,
    axis: int | None = -1,
    *,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
) -> Array: ...
@export
def argsort(
    a: ArrayLike,
    axis: int | None = -1,
    *,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
    dtype: DTypeLike | None = None,
) -> Array: ...
@export
def partition(a: ArrayLike, kth: int, axis: int = -1) -> Array: ...
@export
def argpartition(a: ArrayLike, kth: int, axis: int = -1) -> Array: ...
@export
@api.jit
def sort_complex(a: ArrayLike) -> Array: ...
@export
def lexsort(
    keys: Array | np.ndarray | Sequence[ArrayLike],
    axis: int = -1,
) -> Array: ...
