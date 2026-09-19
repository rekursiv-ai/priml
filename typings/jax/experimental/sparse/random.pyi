from jax import (
    random as random,
    vmap as vmap,
)
from jax._src import dtypes as dtypes
from jax._src.util import split_list as split_list
from jax.experimental import sparse as sparse

def random_bcoo(
    key,
    shape,
    *,
    dtype=...,
    indices_dtype=None,
    nse: float = 0.2,
    n_batch: int = 0,
    n_dense: int = 0,
    unique_indices: bool = True,
    sorted_indices: bool = False,
    generator=...,
    **kwds,
): ...
