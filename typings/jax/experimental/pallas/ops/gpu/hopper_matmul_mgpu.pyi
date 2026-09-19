import dataclasses
import enum

from jax import lax as lax
from jax.experimental.mosaic.gpu import profiler as profiler
from jax.extend import backend as backend

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

def kernel(
    a_gmem,
    b_gmem,
    out_gmem,
    config: TuningConfig,
    pipeline_callback=None,
    delay_release: int = 0,
): ...
def matmul(a, b, config: TuningConfig): ...
def main(_) -> None: ...
