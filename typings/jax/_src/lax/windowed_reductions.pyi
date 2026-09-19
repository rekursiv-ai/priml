from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

from _typeshed import Incomplete
from jax._src import (
    ad_util as ad_util,
    api_util as api_util,
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
    tree_util as tree_util,
    util as util,
)
from jax._src.core import (
    ClosedJaxpr as ClosedJaxpr,
    ShapedArray as ShapedArray,
    jaxpr_as_fun as jaxpr_as_fun,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)
from jax._src.lax import (
    convolution as convolution,
    lax as lax,
    slicing as slicing,
)
from jax._src.lax.other import logaddexp as logaddexp
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import hlo as hlo
from jax._src.typing import Array as Array

map: Incomplete
zip: Incomplete

def reduce_window(
    operand: Any,
    init_value: Any,
    computation: Callable,
    window_dimensions: core.Shape,
    window_strides: Sequence[int] | None = None,
    padding: str | Sequence[tuple[int, int]] = "VALID",
    base_dilation: Sequence[int] | None = None,
    window_dilation: Sequence[int] | None = None,
) -> Any: ...

reduce_window_p: Incomplete

def reduce_window_jvp(
    primals,
    tangents,
    window_dimensions,
    window_strides,
    padding,
    base_dilation,
    window_dilation,
    jaxpr,
    consts,
): ...
def reduce_window_sharding_rule(
    operand,
    window_dimensions,
    window_strides,
    padding,
    base_dilation,
    window_dilation,
): ...

reduce_window_sum_p: Incomplete

def reduce_window_shape_tuple(
    operand_shape,
    window_dimensions,
    window_strides,
    padding,
    base_dilation=None,
    window_dilation=None,
): ...

reduce_window_max_p: Incomplete
reduce_window_min_p: Incomplete
select_and_scatter_p: Incomplete
select_and_scatter_add_p: Incomplete
select_and_gather_add_p: Incomplete
