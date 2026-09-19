from _typeshed import Incomplete
from jax._src import (
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
    state as state,
)
from jax._src.named_sharding import (
    DuplicateSpecError as DuplicateSpecError,
    NamedSharding as NamedSharding,
)
from jax._src.typing import (
    DimSize as DimSize,
    DType as DType,
    Shape as Shape,
)
from jax._src.util import safe_zip as safe_zip

zip: Incomplete
unsafe_zip: Incomplete

def input_dtype(x, *_, **__): ...
def standard_primitive(
    shape_rule,
    dtype_rule,
    name,
    weak_type_rule=None,
    sharding_rule=None,
    vma_rule=None,
    unreduced_rule=None,
    reduced_rule=None,
    memory_space_rule=None,
): ...
def call_reduced_rule(prim, reduced_rule, out_s, num_out, *avals, **kwargs): ...
def call_unreduced_rule(prim, unreduced_rule, out_s, num_out, *avals, **kwargs): ...
def call_sharding_rule(
    prim,
    sh_rule,
    unreduced_rule,
    reduced_rule,
    num_out,
    *avals,
    **kwargs,
): ...
def call_shape_dtype_sharding_rule(
    prim,
    shape_rule,
    dtype_rule,
    sharding_rule,
    unreduced_rule,
    reduced_rule,
    multi_out,
    *avals,
    **kwargs,
): ...
def multi_mem_space_rule(prim, num_out, *avals, **kwargs): ...
def standard_abstract_eval(
    prim,
    shape_rule,
    dtype_rule,
    weak_type_rule,
    sharding_rule,
    vma_rule,
    unreduced_rule,
    reduced_rule,
    memory_space_rule,
    *avals,
    **kwargs,
): ...
def standard_multi_result_abstract_eval(
    prim,
    shape_rule,
    dtype_rule,
    weak_type_rule,
    sharding_rule,
    vma_rule,
    unreduced_rule,
    reduced_rule,
    *avals,
    **kwargs,
): ...
def dtype_to_string(dtype): ...
def int_dtype_for_dim(d: DimSize, *, signed: bool) -> DType: ...
def int_dtype_for_shape(shape: Shape, *, signed: bool) -> DType: ...
def ensure_shaped(
    *avals: core.AbstractValue,
) -> tuple[core.ShapedArray | state.AbstractRef, ...]: ...
