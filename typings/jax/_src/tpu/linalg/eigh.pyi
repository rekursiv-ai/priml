from typing import NamedTuple

from jax._src import (
    api as api,
    config as config,
    core as core,
    dtypes as dtypes,
    lax as lax,
)
from jax._src.interpreters import mlir as mlir
from jax._src.lax import control_flow as control_flow
from jax._src.lax.linalg import is_constant_shape as is_constant_shape
from jax._src.lib.mlir.dialects import hlo as hlo
from jax._src.numpy import (
    reductions as reductions,
    tensor_contractions as tensor_contractions,
    ufuncs as ufuncs,
)
from jax._src.tpu.linalg import qdwh as qdwh
from jax._src.tpu.linalg.stack import Stack as Stack
from jax._src.typing import Array as Array

def split_spectrum(H, n, split_point, V0=None): ...

class _Subproblem(NamedTuple):
    offset: Array
    size: Array

def eigh(
    H,
    *,
    precision: str = "float32",
    termination_size: int = 256,
    n=None,
    sort_eigenvalues: bool = True,
    subset_by_index=None,
): ...
