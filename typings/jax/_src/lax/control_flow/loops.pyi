from collections.abc import Callable as Callable
from typing import Any, TypeVar

from _typeshed import Incomplete
from jax._src import (
    ad_checkpoint as ad_checkpoint,
    ad_util as ad_util,
    api as api,
    api_util as api_util,
    config as config,
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
    effects as effects,
    literals as literals,
    source_info_util as source_info_util,
    state as state,
    util as util,
)
from jax._src.api_util import (
    check_no_aliased_ref_args as check_no_aliased_ref_args,
    check_no_transformed_refs_args as check_no_transformed_refs_args,
)
from jax._src.core import (
    AbstractValue as AbstractValue,
    ClosedJaxpr as ClosedJaxpr,
    ShapedArray as ShapedArray,
    cur_qdd as cur_qdd,
    typeof as typeof,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
    pxla as pxla,
)
from jax._src.lax import (
    lax as lax,
    slicing as slicing,
    windowed_reductions as windowed_reductions,
)
from jax._src.lax.other import logaddexp as logaddexp
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import hlo as hlo
from jax._src.mesh import use_abstract_mesh as use_abstract_mesh
from jax._src.pjit import (
    auto_axes as auto_axes,
    reshard as reshard,
)
from jax._src.sharding_impls import canonicalize_sharding as canonicalize_sharding
from jax._src.state import AbstractRef as AbstractRef
from jax._src.traceback_util import api_boundary as api_boundary
from jax._src.tree_util import (
    FlatTree as FlatTree,
    equality_errors as equality_errors,
    keystr as keystr,
    tree_flatten as tree_flatten,
    tree_map as tree_map,
    tree_unflatten as tree_unflatten,
    treedef_is_leaf as treedef_is_leaf,
)
from jax._src.typing import Array as Array
from jax._src.util import (
    merge_lists as merge_lists,
    partition_list as partition_list,
    safe_map as safe_map,
    safe_zip as safe_zip,
    split_list as split_list,
    split_list_checked as split_list_checked,
    subs_list as subs_list,
    unzip2 as unzip2,
    weakref_lru_cache as weakref_lru_cache,
)

zip = safe_zip
T = TypeVar("T")
type BooleanNumeric = Any
Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")

def scan(
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X | None = None,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
) -> tuple[Carry, Y]: ...

eval_jaxpr_p: Incomplete
scan_p: Incomplete

def while_loop(
    cond_fun: Callable[[T], BooleanNumeric],
    body_fun: Callable[[T], T],
    init_val: T,
) -> T: ...

while_p: Incomplete

def fori_loop(
    lower,
    upper,
    body_fun,
    init_val,
    *,
    unroll: int | bool | None = None,
): ...
@api_boundary
def map(f, xs, *, batch_size: int | None = None): ...
@api_boundary
def associative_scan(fn: Callable, elems, reverse: bool = False, axis: int = 0): ...
def cumsum(operand: Array, axis: int = 0, reverse: bool = False) -> Array: ...
def cumprod(operand: Array, axis: int = 0, reverse: bool = False) -> Array: ...
def cummax(operand: Array, axis: int = 0, reverse: bool = False) -> Array: ...
def cummin(operand: Array, axis: int = 0, reverse: bool = False) -> Array: ...
def cumlogsumexp(operand: Array, axis: int = 0, reverse: bool = False) -> Array: ...
def cumred_reduce_window_impl(
    window_reduce: Callable,
    x,
    *,
    axis: int,
    reverse: bool,
): ...

cumsum_p: Incomplete
cumlogsumexp_p: Incomplete
cumprod_p: Incomplete
cummax_p: Incomplete
cummin_p: Incomplete
