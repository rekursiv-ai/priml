from _typeshed import Incomplete
from jax._src import (
    core as core,
    dispatch as dispatch,
)
from jax._src.api_util import debug_info as debug_info
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
    partial_eval as pe,
)
from jax._src.lib.mlir import ir as ir
from jax._src.tree_util import (
    FlatTree as FlatTree,
    tree_flatten as tree_flatten,
    tree_unflatten as tree_unflatten,
)
from jax._src.util import (
    safe_map as safe_map,
    safe_zip as safe_zip,
    split_list as split_list,
    unzip2 as unzip2,
    weakref_lru_cache as weakref_lru_cache,
)

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete

def scheduling_group(name): ...
def xla_metadata_call(f=None, **meta): ...

xla_metadata_call_p: Incomplete

def attr_get(x): ...
def dce_jaxpr_xla_metadata_rule(
    used_outputs: list[bool],
    eqn: pe.JaxprEqn,
) -> tuple[list[bool], pe.JaxprEqn | None]: ...
