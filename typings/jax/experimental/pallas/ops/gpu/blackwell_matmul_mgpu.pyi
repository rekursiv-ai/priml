import dataclasses
import enum

from jax import lax as lax
from jax.experimental.mosaic.gpu import profiler as profiler

class MatmulDimension(enum.IntEnum):
    M = 0
    N = 1

@dataclasses.dataclass(frozen=True)
class TuningConfig:
    tile_m: int
    tile_n: int
    tile_k: int
    max_concurrent_steps: int
    collective: bool
    epilogue_tile_n: int = ...
    grid_minor_dim: MatmulDimension = ...
    grid_tile_width: int = ...

def matmul_kernel(a, b, config: TuningConfig): ...
def main(_) -> None: ...
