from _typeshed import Incomplete
from jax._src import (
    core as core,
    dispatch as dispatch,
)
from jax._src.lax.lax import DotDimensionNumbers as DotDimensionNumbers
from jax._src.lib.mlir.dialects import mhlo as mhlo
from jax._src.typing import (
    Array as Array,
    DTypeLike as DTypeLike,
)
from jax.interpreters import mlir as mlir

nm_spmm_p: Incomplete

def nm_spmm(
    lhs: Array,
    rhs: Array,
    metadata: Array,
    dimension_numbers: DotDimensionNumbers = ...,
    sparse_operand_idx: int = 0,
    output_dtype: DTypeLike = ...,
) -> Array: ...

nm_pack_p: Incomplete

def nm_pack(mask: Array, n: int = 2, m: int = 4) -> Array: ...
