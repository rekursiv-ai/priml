from jax import lax as lax
from jax.experimental import multihost_utils as multihost_utils
from jax.experimental.mosaic.gpu import profiler as profiler
from jax.experimental.pallas.ops.gpu import hopper_matmul_mgpu as hopper_matmul_mgpu

import jax
import jax.numpy as jnp

MatmulDimension = hopper_matmul_mgpu.MatmulDimension
TuningConfig = hopper_matmul_mgpu.TuningConfig

def is_nvshmem_used() -> bool: ...
def all_gather_lhs_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    axis_name,
    *,
    config: hopper_matmul_mgpu.TuningConfig,
    dtype: jnp.dtype = ...,
) -> jax.Array: ...
