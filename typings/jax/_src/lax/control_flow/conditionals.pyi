from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

from _typeshed import Incomplete
from jax._src import (
    ad_util as ad_util,
    api_util as api_util,
    config as config,
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
    effects as effects,
    source_info_util as source_info_util,
    util as util,
)
from jax._src.core import (
    cur_qdd as cur_qdd,
    replace_jaxpr_effects as replace_jaxpr_effects,
    typeof as typeof,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
    pxla as pxla,
)
from jax._src.lax import lax as lax
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import hlo as hlo
from jax._src.state.discharge import (
    discharge_state as discharge_state,
    register_partial_discharge_rule as register_partial_discharge_rule,
)
from jax._src.state.types import (
    AbstractRef as AbstractRef,
    RefEffect as RefEffect,
)
from jax._src.traceback_util import api_boundary as api_boundary
from jax._src.tree_util import (
    FlatTree as FlatTree,
    equality_errors_pytreedef as equality_errors_pytreedef,
    keystr as keystr,
    tree_flatten as tree_flatten,
    tree_flatten_with_path as tree_flatten_with_path,
    tree_unflatten as tree_unflatten,
)
from jax._src.typing import ArrayLike as ArrayLike
from jax._src.util import (
    partition_list as partition_list,
    safe_map as safe_map,
    safe_zip as safe_zip,
    split_list as split_list,
    unzip2 as unzip2,
)

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete

@api_boundary
def switch(index, branches: Sequence[Callable], *operands: Any, operand: Any = ...): ...
def cond(pred, true_fun: Callable, false_fun: Callable, *operands, operand=...): ...

type BranchesPlatforms = tuple[tuple[str, ...] | None, ...]
cond_p: Incomplete

def platform_dependent(
    *args: Any,
    default: Callable[..., _T] | None = None,
    **per_platform: Callable[..., _T],
): ...

platform_index_p: Incomplete
