from collections.abc import Sequence
from typing import NamedTuple

from _typeshed import Incomplete
from jax._src import (
    core as core,
    dtypes as dtypes,
    util as util,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)
from jax._src.lax import lax as lax
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import hlo as hlo
from jax._src.sharding_impls import (
    NamedSharding as NamedSharding,
    PartitionSpec as P,
    canonicalize_sharding as canonicalize_sharding,
)
from jax._src.typing import (
    Array as Array,
    DTypeLike as DTypeLike,
)

class ConvDimensionNumbers(NamedTuple):
    lhs_spec: Sequence[int]
    rhs_spec: Sequence[int]
    out_spec: Sequence[int]

type ConvGeneralDilatedDimensionNumbers = (
    tuple[str, str, str] | ConvDimensionNumbers | None
)

def conv_general_dilated(
    lhs: Array,
    rhs: Array,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: ConvGeneralDilatedDimensionNumbers = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    precision: lax.PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    out_sharding: NamedSharding | P | None = None,
) -> Array: ...
def conv(
    lhs: Array,
    rhs: Array,
    window_strides: Sequence[int],
    padding: str,
    precision: lax.PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array: ...
def conv_with_general_padding(
    lhs: Array,
    rhs: Array,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int] | None,
    rhs_dilation: Sequence[int] | None,
    precision: lax.PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array: ...
def conv_transpose(
    lhs: Array,
    rhs: Array,
    strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: ConvGeneralDilatedDimensionNumbers = None,
    transpose_kernel: bool = False,
    precision: lax.PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    use_consistent_padding: bool = False,
) -> Array: ...

conv_general_dilated_p: Incomplete

def conv_shape_tuple(
    lhs_shape,
    rhs_shape,
    strides,
    pads,
    batch_group_count: int = 1,
): ...
def conv_general_shape_tuple(
    lhs_shape,
    rhs_shape,
    window_strides,
    padding,
    dimension_numbers,
): ...
def conv_transpose_shape_tuple(
    lhs_shape,
    rhs_shape,
    window_strides,
    padding,
    dimension_numbers,
): ...
def conv_dimension_numbers(
    lhs_shape,
    rhs_shape,
    dimension_numbers,
) -> ConvDimensionNumbers: ...
def conv_general_permutations(dimension_numbers): ...
