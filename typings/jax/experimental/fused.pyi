from _typeshed import Incomplete
from jax._src import (
    core as core,
    dispatch as dispatch,
)
from jax._src.api_util import (
    debug_info as debug_info,
    flatten_fun_nokwargs as flatten_fun_nokwargs,
)
from jax._src.core import typeof as typeof
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)
from jax._src.lib.mlir import ir as ir
from jax._src.tree_util import (
    tree_flatten as tree_flatten,
    tree_unflatten as tree_unflatten,
)
from jax._src.util import (
    safe_map as safe_map,
    safe_zip as safe_zip,
    unzip2 as unzip2,
    weakref_lru_cache as weakref_lru_cache,
)

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete

def fused(*, out_spaces): ...

fused_p: Incomplete
