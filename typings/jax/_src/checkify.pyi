from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import dataclasses

from _typeshed import Incomplete
from jax._src import (
    ad_checkpoint as ad_checkpoint,
    api as api,
    api_util as api_util,
    callback as callback,
    config as config,
    core as core,
    custom_derivatives as custom_derivatives,
    dtypes as dtypes,
    effects as effects,
    lax as lax,
    linear_util as lu,
    pjit as pjit,
    sharding_impls as sharding_impls,
    source_info_util as source_info_util,
    traceback_util as traceback_util,
)
from jax._src.ad_util import SymbolicZero as SymbolicZero
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)
from jax._src.tree_util import (
    FlatTree as FlatTree,
    tree_flatten as tree_flatten,
    tree_map as tree_map,
    tree_unflatten as tree_unflatten,
)
from jax._src.typing import Array as Array
from jax._src.util import (
    HashableWrapper as HashableWrapper,
    as_hashable_function as as_hashable_function,
    foreach as foreach,
    safe_map as safe_map,
    safe_zip as safe_zip,
    split_list as split_list,
    unzip3 as unzip3,
    weakref_lru_cache as weakref_lru_cache,
)

import numpy as np

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
type Bool = bool | Array
type Int = int | Array
ErrorCategory: Incomplete
type Payload = list[np.ndarray | Array]
PyTreeDef: Incomplete
Out = TypeVar("Out")

class JaxException(Exception):
    traceback_info: Incomplete
    def __init__(self, traceback_info) -> None: ...
    def __init_subclass__(cls) -> None: ...
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, metadata, payload): ...
    def get_effect_type(self) -> ErrorEffect: ...

@dataclasses.dataclass(eq=True, frozen=True)
class ErrorEffect(effects.Effect):
    error_type: type[JaxException]
    shape_dtypes: tuple[api.ShapeDtypeStruct, ...]
    def __lt__(self, other: ErrorEffect): ...

class DivisionByZeroError(JaxException):
    def get_effect_type(self): ...

class NaNError(JaxException):
    prim: Incomplete
    def __init__(self, traceback_info, primitive_name) -> None: ...
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, metadata, _): ...
    def get_effect_type(self): ...

class OOBError(JaxException):
    prim: Incomplete
    operand_shape: Incomplete
    def __init__(
        self,
        traceback_info,
        primitive_name,
        operand_shape,
        payload,
    ) -> None: ...
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, metadata, payload): ...
    def get_effect_type(self): ...

class FailedCheckError(JaxException):
    fmt_string: Incomplete
    args: Incomplete
    kwargs: Incomplete
    def __init__(self, traceback_info, fmt_string, *a, **k) -> None: ...
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, metadata, payload): ...
    def get_effect_type(self): ...

@dataclasses.dataclass
class BatchedError(JaxException):
    error_mapping: dict[tuple[int, ...], JaxException]
    def __post_init__(self) -> None: ...

@dataclasses.dataclass(frozen=True)
class Error:
    def get(self) -> str | None: ...
    def get_exception(self) -> JaxException | None: ...
    def throw(self) -> None: ...
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, metadata, data): ...

init_error: Incomplete
next_code: Incomplete

def assert_func(error: Error, pred: Bool, new_error: JaxException) -> Error: ...
def update_error(error, pred, code, metadata, payload, effect_type): ...
def default_checkify_rule(
    primitive: core.Primitive,
    error: Error,
    enabled_errors,
    *invals: core.Value,
    **params: Any,
) -> tuple[Error, Sequence[core.Value]]: ...
def checkify_jaxpr(
    jaxpr: core.ClosedJaxpr,
    enabled_errors,
    error: Error,
    *args,
) -> tuple[Error, list[core.Value]]: ...
def checkify_jaxpr_flat(
    jaxpr: core.Jaxpr,
    consts: Sequence[core.Value],
    enabled_errors,
    err_tree: PyTreeDef,
    *args: core.Value,
) -> tuple[Error, list[Any]]: ...
def checkify_jaxpr_flat_hashable(
    jaxpr,
    hashable_consts,
    enabled_errors,
    err_tree,
    *args,
): ...
@lu.transformation_with_aux2
def flatten_fun_output(f, store, *args): ...

check_p: Incomplete

class JaxRuntimeError(ValueError): ...

@check_p.def_impl
def check_impl(*args, err_tree, debug): ...
@check_p.def_effectful_abstract_eval
def check_abstract_eval(*args, err_tree, debug): ...

functionalization_error: Incomplete

def check_lowering_rule(ctx, *args, err_tree, debug): ...
def check_lowering_rule_unsupported(*a, debug, **k): ...
def python_err(err_tree, *args): ...
def check_batching_rule(batched_args, batch_dims, *, err_tree, debug): ...
def check_jvp_rule(primals, _, *, err_tree, debug): ...

ErrorCheckRule = Callable
error_checks: dict[core.Primitive, ErrorCheckRule]

def get_traceback(): ...
def nan_error_check(prim, error, enabled_errors, *in_vals, **params): ...
def check_nans(prim, error, enabled_errors, out): ...

nan_primitives: Incomplete

def dynamic_slice_error_check(
    error,
    enabled_errors,
    operand,
    *start_indices,
    slice_sizes,
): ...
def dynamic_update_slice_error_check(
    error,
    enabled_errors,
    operand,
    update,
    *start_indices,
): ...
def gather_error_check(
    error,
    enabled_errors,
    operand,
    start_indices,
    *,
    dimension_numbers,
    slice_sizes,
    unique_indices,
    indices_are_sorted,
    mode,
    fill_value,
): ...
def div_error_check(error, enabled_errors, x, y): ...
def oob_payload(oob_mask, indices, dims_map, operand_shape): ...
def scatter_oob(operand, indices, updates, dnums): ...
def scatter_error_check(
    prim,
    error,
    enabled_errors,
    operand,
    indices,
    updates,
    *,
    update_jaxpr,
    update_consts,
    dimension_numbers,
    indices_are_sorted,
    unique_indices,
    mode,
): ...

class ErrorEffects:
    val: Incomplete
    def __init__(self, val) -> None: ...

@weakref_lru_cache
def jaxpr_to_checkify_jaxpr(
    jaxpr: core.ClosedJaxpr,
    enabled_errors,
    err_tree: PyTreeDef,
    *flat_err_and_in_vals,
) -> tuple[core.ClosedJaxpr, PyTreeDef, set[ErrorEffect]]: ...
def cond_error_check(error: Error, enabled_errors, index, *ops, branches, **params): ...
def scan_error_check(
    error,
    enabled_errors,
    *in_flat,
    reverse,
    length,
    jaxpr,
    num_consts,
    num_carry,
    linear,
    unroll,
    _split_transpose,
): ...
def checkify_while_body_jaxpr(
    cond_jaxpr: core.ClosedJaxpr,
    body_jaxpr: core.ClosedJaxpr,
    enabled_errors,
    error: Error,
    c_consts_num: int,
) -> tuple[core.ClosedJaxpr, PyTreeDef, set[ErrorEffect]]: ...
@weakref_lru_cache
def ignore_error_output_jaxpr(jaxpr, num_error_vals: int): ...
def while_loop_error_check(
    error,
    enabled_errors,
    *in_flat,
    cond_nconsts,
    cond_jaxpr,
    body_nconsts,
    body_jaxpr,
): ...
def pjit_error_check(
    error,
    enabled_errors,
    *vals_in,
    jaxpr,
    in_shardings,
    out_shardings,
    in_layouts,
    out_layouts,
    donated_invars,
    ctx_mesh,
    name,
    inline,
    keep_unused,
    compiler_options_kvs,
): ...
def remat_error_check(error, enabled_errors, *vals_in, jaxpr, **params): ...
def shard_map_error_check(
    error: Error,
    enabled_errors,
    *vals_in,
    jaxpr: core.Jaxpr,
    in_specs,
    out_specs,
    **kwargs,
): ...
def custom_jvp_call_rule(
    in_err: Error,
    enabled_errors: set,
    *in_vals,
    num_consts,
    jvp_jaxpr_fun: lu.WrappedFun,
    call_jaxpr: core.ClosedJaxpr,
    **params,
): ...
def lift_jvp(
    num_errs: int,
    num_consts: int,
    jvp_jaxpr_fun: lu.WrappedFun,
) -> lu.WrappedFun: ...
def custom_vjp_call_rule(
    in_err,
    enabled_errors,
    *in_vals,
    call_jaxpr: core.ClosedJaxpr,
    fwd_jaxpr_thunk,
    num_consts,
    bwd: lu.WrappedFun,
    out_trees,
    symbolic_zeros: bool,
): ...
def check_discharge_rule(error, enabled_errors, *args, err_tree, debug): ...

user_checks: Incomplete
nan_checks: Incomplete
index_checks: Incomplete
div_checks: Incomplete
type float_checks = nan_checks | div_checks
type automatic_checks = float_checks | index_checks
type all_checks = automatic_checks | user_checks

def checkify(
    f: Callable[..., Out],
    errors: frozenset[ErrorCategory] = ...,
) -> Callable[..., tuple[Error, Out]]: ...
def check(
    pred: Bool,
    msg: str,
    *fmt_args,
    debug: bool = False,
    **fmt_kwargs,
) -> None: ...
def is_scalar_pred(pred) -> bool: ...
def debug_check(pred: Bool, msg: str, *fmt_args, **fmt_kwargs) -> None: ...
def check_error(error: Error) -> None: ...
