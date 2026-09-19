from collections.abc import Callable

from _typeshed import Incomplete
from jax._src import (
    api_util as api_util,
    core as core,
    source_info_util as source_info_util,
)
from jax._src.tree_util import (
    FlatTree as FlatTree,
    Partial as Partial,
    tree_leaves_checked as tree_leaves_checked,
    tree_unflatten as tree_unflatten,
)
from jax._src.util import (
    safe_map as safe_map,
    safe_zip as safe_zip,
    unzip2 as unzip2,
    weakref_lru_cache as weakref_lru_cache,
)

map = safe_map
zip = safe_zip

def remat_transform(policy, f, *args): ...

class RematTracer(core.Tracer):
    val: Incomplete
    tracer: Incomplete
    def __init__(self, trace, x, jaxpr_tracer) -> None: ...
    @property
    def aval(self): ...

class RematTrace(core.Trace):
    parent_trace: Incomplete
    jaxpr_trace: Incomplete
    tag: Incomplete
    policy: Incomplete
    requires_low: bool
    def __init__(self, parent_trace, jaxpr_trace, tag, policy) -> None: ...
    def to_val_tracer_pair(self, x): ...
    def process_primitive(self, prim, tracers, params): ...

def reduce_precision(x): ...

rules: dict[core.Primitive, Callable]
reduce_precision_handlers: dict[type, Callable]

def remat_jaxpr(jaxpr, policy): ...
