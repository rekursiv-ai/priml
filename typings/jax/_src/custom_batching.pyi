from collections.abc import Callable as Callable
from typing import Any

from _typeshed import Incomplete
from jax._src import (
    api as api,
    api_util as api_util,
    core as core,
    custom_api_util as custom_api_util,
    linear_util as lu,
    source_info_util as source_info_util,
    traceback_util as traceback_util,
    tree_util as tree_util,
    util as util,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
    pxla as pxla,
)
from jax._src.interpreters.batching import not_mapped as not_mapped
from jax._src.tree_util import (
    tree_flatten as tree_flatten,
    tree_map as tree_map,
    tree_structure as tree_structure,
    tree_unflatten as tree_unflatten,
    treedef_tuple as treedef_tuple,
)

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete

class custom_vmap:
    fun: Callable[..., Any]
    vmap_rule: Callable[..., tuple[Any, Any]] | None
    def __init__(self, fun: Callable[..., Any]) -> None: ...
    __getattr__: Incomplete
    def def_vmap(
        self,
        vmap_rule: Callable[..., tuple[Any, Any]],
    ) -> Callable[..., tuple[Any, Any]]: ...
    @traceback_util.api_boundary
    def __call__(self, *args, **kwargs): ...

class ClosedRule:
    rule: Incomplete
    debug: Incomplete
    def __init__(self, rule: Callable, debug: core.DebugInfo) -> None: ...
    def __call__(self, axis_size, all_in_batched, *all_args): ...

def ensure_list(xs): ...
def rule_name(rule): ...
def call_rule(rule, axis_size, in_batched, args): ...
def check_vmap_rule_trees(
    rule,
    original_out_tree,
    out_tree,
    out_batched_tree,
) -> None: ...
def maybe_bdim_at_front(x, bdim): ...
def vmap_unrestricted(f: lu.WrappedFun, *args, in_axes, axis_name, axis_size): ...
def custom_vmap_impl(*args, call, rule, in_tree, out_tree): ...
def custom_vmap_batching(args_flat, dims, *, call, rule, in_tree, out_tree): ...
def custom_vmap_abstract_eval(*in_avals, call, **_): ...
def custom_vmap_jvp(
    primals,
    tangents,
    *,
    call: core.ClosedJaxpr,
    rule: ClosedRule,
    in_tree: tree_util.PyTreeDef,
    out_tree: tree_util.PyTreeDef,
): ...

custom_vmap_p: Incomplete

def tree_split(mask, tree): ...
def tree_merge(mask, lhs_tree, rhs_tree): ...
def sequential_vmap(f): ...
