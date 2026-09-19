from collections.abc import Sequence
from typing import Any

from jax._src import (
    dtypes as dtypes,
    util as util,
)
from jax._src.custom_derivatives import custom_jvp as custom_jvp
from jax._src.lax import (
    convolution as convolution,
    lax as lax,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)

type DType = Any

def conv_general_dilated_patches(
    lhs: ArrayLike,
    filter_shape: Sequence[int],
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: convolution.ConvGeneralDilatedDimensionNumbers | None = None,
    precision: lax.Precision | None = None,
    preferred_element_type: DType | None = None,
) -> Array: ...
def conv_general_dilated_local(
    lhs: ArrayLike,
    rhs: ArrayLike,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    filter_shape: Sequence[int],
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: convolution.ConvGeneralDilatedDimensionNumbers | None = None,
    precision: lax.PrecisionLike = None,
) -> Array: ...
@custom_jvp
def logaddexp(x1: ArrayLike, x2: ArrayLike, /) -> Array: ...
@custom_jvp
def logaddexp2(x1: ArrayLike, x2: ArrayLike, /) -> Array: ...
