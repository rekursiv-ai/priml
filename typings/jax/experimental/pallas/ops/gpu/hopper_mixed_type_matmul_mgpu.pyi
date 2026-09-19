import dataclasses
import enum

from jax import lax as lax
from jax._src import dtypes as dtypes
from jax.experimental.mosaic.gpu import profiler as profiler
from jax.extend import backend as backend

import jax
import jax.numpy as jnp

class MatmulDimension(enum.IntEnum):
    M = 0
    N = 1

@dataclasses.dataclass(frozen=True)
class TuningConfig:
    tile_m: int
    tile_n: int
    tile_k: int
    max_concurrent_steps: int
    epi_tile_n: int | None = ...
    epi_tile_m: int | None = ...
    grid_minor_dim: MatmulDimension = ...
    grid_tile_width: int = ...
    wg_dimension: MatmulDimension = ...
    cluster_dimension: MatmulDimension | None = ...

def mixed_matmul_kernel(
    a: jax.Array,
    b: jax.Array,
    *,
    out_dtype: jnp.dtype,
    config: TuningConfig,
) -> jax.Array: ...
def reference(a: jax.Array, b: jax.Array, *, out_dtype: jnp.dtype) -> jax.Array: ...
def main(_) -> None: ...
