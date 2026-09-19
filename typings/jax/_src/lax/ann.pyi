from _typeshed import Incomplete
from jax._src import (
    ad_util as ad_util,
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import (
    func as func,
    hlo as hlo,
)
from jax._src.numpy.indexing import take_along_axis as take_along_axis
from jax._src.typing import Array as Array

def approx_max_k(
    operand: Array,
    k: int,
    reduction_dimension: int = -1,
    recall_target: float = 0.95,
    reduction_input_size_override: int = -1,
    aggregate_to_topk: bool = True,
) -> tuple[Array, Array]: ...
def approx_min_k(
    operand: Array,
    k: int,
    reduction_dimension: int = -1,
    recall_target: float = 0.95,
    reduction_input_size_override: int = -1,
    aggregate_to_topk: bool = True,
) -> tuple[Array, Array]: ...

approx_top_k_p: Incomplete
