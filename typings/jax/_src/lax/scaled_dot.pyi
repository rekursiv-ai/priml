from _typeshed import Incomplete
from jax._src import (
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
)
from jax._src.interpreters import (
    batching as batching,
    mlir as mlir,
)
from jax._src.lax import lax as lax
from jax._src.typing import (
    Array as Array,
    DTypeLike as DTypeLike,
)

scaled_dot_p: Incomplete
scaled_dot_lowering: Incomplete

def scaled_dot(
    lhs: Array,
    rhs: Array,
    *,
    lhs_scale: Array | None = None,
    rhs_scale: Array | None = None,
    dimension_numbers: lax.DotDimensionNumbers | None = None,
    preferred_element_type: DTypeLike | None = None,
): ...
