from contextlib import contextmanager

from _typeshed import Incomplete
from jax._src import (
    config as config,
    core as core,
    dispatch as dispatch,
)
from jax._src.api_util import (
    debug_info as debug_info,
    flatten_axes as flatten_axes,
    flatten_fun_nokwargs as flatten_fun_nokwargs,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)
from jax._src.lib import xla_client as xla_client
from jax._src.lib.mlir import ir as ir
from jax._src.tree_util import (
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

config_ext: Incomplete
map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete

@contextmanager
def extend_compute_type(c_type: str | None): ...
@contextmanager
def compute_on(compute_type: str): ...
def compute_on2(f=None, *, compute_type, out_memory_spaces): ...

compute_on_p: Incomplete
