from collections.abc import Sequence

import dataclasses

from jax import lax as lax
from jax.experimental.mosaic.gpu import profiler as profiler
from jax.experimental.pallas.ops.gpu import (
    blackwell_matmul_mgpu as blackwell_matmul_mgpu,
    ragged_dot_mgpu as ragged_dot_mgpu,
)

import jax

@dataclasses.dataclass(frozen=True)
class TuningConfig:
    tile_m: int
    tile_n: int
    tile_k: int
    max_concurrent_steps: int
    collective: bool
    grid_tile_width: int
    grid_minor_dim: blackwell_matmul_mgpu.MatmulDimension
    epilogue_tile_n: int = ...

def do_matmul(
    a_gmem,
    b_gmem,
    out_gmem,
    grid_indices: Sequence[jax.Array],
    wg_axis: str,
    collective_axes: tuple[str, ...],
    local_index: jax.Array,
    config: TuningConfig,
    group_info: ragged_dot_mgpu.GroupInfo,
    a_smem,
    b_smem,
    acc_tmem,
    acc_smem,
    a_tma_barrier,
    b_tma_barrier,
    store_done_barrier,
    mma_done_barrier,
    consumed_barrier,
): ...
def ragged_dot_kernel(a, b, group_sizes, config: TuningConfig): ...
def ragged_dot_reference(a, b, g): ...
def sample_group_sizes(
    key: jax.Array,
    num_groups: int,
    num_elements: int,
    alpha: float = 10.0,
): ...
def main(_) -> None: ...
