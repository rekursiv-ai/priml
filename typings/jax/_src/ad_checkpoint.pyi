from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

from _typeshed import Incomplete
from jax._src import (
    ad_util as ad_util,
    api as api,
    api_util as api_util,
    config as config,
    core as core,
    custom_derivatives as custom_derivatives,
    deprecations as deprecations,
    dtypes as dtypes,
    effects as effects,
    source_info_util as source_info_util,
    traceback_util as traceback_util,
)
from jax._src.hijax import VJPHiPrimitive as VJPHiPrimitive
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
    partial_eval as pe,
)
from jax._src.interpreters.remat import remat_transform as remat_transform
from jax._src.lib.mlir.dialects import hlo as hlo
from jax._src.state import discharge as discharge
from jax._src.state.types import AbstractRef as AbstractRef
from jax._src.traceback_util import api_boundary as api_boundary
from jax._src.tree_util import (
    PyTreeDef as PyTreeDef,
    broadcast_prefix as broadcast_prefix,
    tree_flatten as tree_flatten,
    tree_map as tree_map,
    tree_structure as tree_structure,
    tree_unflatten as tree_unflatten,
)
from jax._src.typing import DeprecatedArg as DeprecatedArg
from jax._src.util import (
    merge_lists as merge_lists,
    partition_list as partition_list,
    safe_map as safe_map,
    safe_zip as safe_zip,
    split_list as split_list,
    unzip2 as unzip2,
    weakref_lru_cache as weakref_lru_cache,
    wraps as wraps,
)

map = safe_map
zip = safe_zip
logger: Incomplete

def everything_saveable(*_, **__) -> bool: ...
def nothing_saveable(*_, **__) -> bool: ...
def dots_saveable(prim, *_, **__) -> bool: ...

checkpoint_dots = dots_saveable

def dots_with_no_batch_dims_saveable(prim, *args, **params) -> bool: ...
def offload_dot_with_no_batch_dims(offload_src, offload_dst): ...

name_p: Incomplete

def save_anything_except_these_names(*names_not_to_save): ...
def save_any_names_but_these(*names_not_to_save): ...
def save_only_these_names(*names_which_can_be_saved): ...
def save_and_offload_only_these_names(
    *,
    names_which_can_be_saved,
    names_which_can_be_offloaded,
    offload_src,
    offload_dst,
): ...
def save_from_both_policies(policy_1, policy_2): ...

checkpoint_policies: Incomplete

def checkpoint(
    fun: Callable,
    *,
    prevent_cse: bool | Sequence[bool] = True,
    policy: Callable[..., bool] | None = None,
    static_argnums: int | tuple[int, ...] = (),
    concrete: bool | DeprecatedArg = ...,
) -> Callable: ...
def remat(
    fun: Callable,
    *,
    prevent_cse: bool = True,
    policy: Callable[..., bool] | None = None,
    static_argnums: int | tuple[int, ...] = (),
    concrete: bool | DeprecatedArg = ...,
) -> Callable: ...

class WrapHashably:
    val: Any
    hash: int
    hashable: bool
    def __init__(self, val) -> None: ...
    def __hash__(self): ...
    def __eq__(self, other): ...

def saved_residuals(
    f: Callable,
    *args,
    **kwargs,
) -> list[tuple[core.AbstractValue, str]]: ...
def print_saved_residuals(f, *args, **kwargs) -> None: ...

remat_p: Incomplete

@remat_p.def_impl
def remat_impl(*args, jaxpr, prevent_cse, differentiated, policy): ...
@remat_p.def_effectful_abstract_eval
def remat_abstract_eval(*args, jaxpr, prevent_cse, differentiated, policy): ...
def remat_jvp(primals, tangents, jaxpr, prevent_cse, differentiated, policy): ...
def remat_partial_eval(
    trace: pe.JaxprTrace,
    *tracers: core.Tracer,
    jaxpr: core.Jaxpr,
    prevent_cse,
    **params,
): ...
def remat_partial_eval_custom_params_updater(*args): ...
def remat_transpose(out_cts, *args, jaxpr, prevent_cse, **params) -> None: ...
def transpose_jaxpr(
    jaxpr: core.ClosedJaxpr,
    in_linear: bool | Sequence[bool],
    out_zeros: bool | Sequence[bool],
) -> tuple[core.ClosedJaxpr, list[bool]]: ...
def remat_vmap(axis_data, args, dims, *, jaxpr, **params): ...
def remat_dce(
    used_outputs: list[bool],
    eqn: core.JaxprEqn,
) -> tuple[list[bool], core.JaxprEqn | None]: ...
def remat_expansion(
    *args,
    jaxpr: core.Jaxpr,
    prevent_cse: bool,
    differentiated: bool,
    **_,
): ...
def checkpoint_name(x, name): ...
def name_jvp(primals, tangents, *, name): ...
def name_batcher(args, dims, *, name): ...
def checkpoint_name3(name, x): ...
def remat3(f=None, /, policy=...): ...

class RematTraced(VJPHiPrimitive):
    traced: Any
    policy: Any
    out_aval: Incomplete
    params: Incomplete
    def __init__(self, traced, policy) -> None: ...
    def expand(self, *args): ...
    def vjp_fwd(self, _nzs_in, *primals): ...
    def vjp_bwd(self, res, outgrad, *arg_accums) -> None: ...
    def jvp(self, primals, tangents): ...
    def lin(self, nzs_in, *primals): ...
    def linearized(self, primals, *tangents): ...

class CheckpointName(VJPHiPrimitive):
    in_avals: Incomplete
    out_aval: Incomplete
    params: Incomplete
    def __init__(self, name, aval) -> None: ...
    def expand(self, x): ...
    def remat(self, policy, x): ...

@custom_derivatives.custom_jvp
def primal_left_tangent_right(x, _x): ...
