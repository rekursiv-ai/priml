from collections.abc import Callable as Callable

from _typeshed import Incomplete
from jax._src import (
    api as api,
    core as core,
    dtypes as dtypes,
)
from jax._src.lax import lax as lax
from jax._src.state.primitives import ref_get as ref_get
from jax._src.state.types import AbstractRef as AbstractRef
from jax._src.typing import DTypeLike as DTypeLike
from jax._src.util import (
    safe_map as safe_map,
    safe_zip as safe_zip,
    split_list as split_list,
)

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete

def hoist_consts_to_refs(
    jaxpr: core.Jaxpr,
    *,
    index: int = 0,
    make_abstract_ref: Callable[[core.AbstractValue], AbstractRef] = ...,
) -> core.Jaxpr: ...
def bitcast(x, dtype: DTypeLike): ...
def eval_bitcast_shape(x, dtype: DTypeLike): ...
