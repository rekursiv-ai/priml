from collections.abc import Callable as Callable
from typing import Any

from _typeshed import Incomplete
from jax._src import (
    ad_util as ad_util,
    api_util as api_util,
    core as core,
    custom_api_util as custom_api_util,
    linear_util as lu,
    source_info_util as source_info_util,
    traceback_util as traceback_util,
    util as util,
)
from jax._src.interpreters import (
    ad as ad,
    mlir as mlir,
    pxla as pxla,
)
from jax._src.tree_util import (
    PyTreeDef as PyTreeDef,
    tree_flatten as tree_flatten,
    tree_leaves as tree_leaves,
    tree_map as tree_map,
    tree_structure as tree_structure,
    tree_unflatten as tree_unflatten,
    treedef_tuple as treedef_tuple,
)

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete

class StoreEqual(lu.Store):
    def store(self, val) -> None: ...

@util.curry
def transformation_with_aux(
    gen,
    fun: lu.WrappedFun,
    *gen_static_args,
) -> tuple[lu.WrappedFun, Any]: ...

flatten_fun_nokwargs: Incomplete

class custom_transpose:
    fun: Callable
    transpose: Callable | None
    def __init__(self, fun: Callable) -> None: ...
    __getattr__: Incomplete
    def def_transpose(self, transpose: Callable): ...
    @traceback_util.api_boundary
    def __call__(self, out_types, res_arg, lin_arg): ...

def tree_fill(x, treedef): ...
def tree_fill_like(x, tree): ...
def tree_broadcast(full_treedef, tree, is_leaf=None): ...
def is_treedef_prefix(entire, prefix): ...
def rule_name(rule): ...
def check_transpose_rule_trees(
    rule: lu.WrappedFun,
    lin_tree: PyTreeDef,
    rule_out_tree: PyTreeDef,
): ...
def make_transpose_from_thunk(
    thunk: Callable,
    lin_tree: PyTreeDef,
) -> lu.WrappedFun: ...

class CustomTransposePrimitive(core.Primitive):
    call_primitive: bool
    map_primitive: bool
    multiple_results: bool
    def bind(self, *args, **params): ...
    def bind_with_trace(self, trace, call_args, params): ...
    def get_bind_params(self, params): ...

def custom_transpose_typecheck(_, *in_atoms, out_types, **params): ...
def custom_transpose_transpose_rule(
    cts,
    *args,
    out_types,
    res_tree,
    lin_tree,
    out_tree,
    **params,
): ...
def custom_transpose_lowering(*args, call_jaxpr, **params): ...

custom_transpose_p: Incomplete
