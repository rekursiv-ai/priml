from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

from _typeshed import Incomplete
from jax._src import (
    api_util as api_util,
    config as config,
    core as core,
    linear_util as lu,
    source_info_util as source_info_util,
)
from jax._src.ad_util import (
    SymbolicZero as SymbolicZero,
    Zero as Zero,
    add_jaxvals as add_jaxvals,
    add_jaxvals_p as add_jaxvals_p,
    p2cz as p2cz,
    p2tz as p2tz,
    replace_internal_symbolic_zeros as replace_internal_symbolic_zeros,
    replace_rule_output_symbolic_zeros as replace_rule_output_symbolic_zeros,
    zeros_like_aval as zeros_like_aval,
)
from jax._src.api_util import (
    debug_info as debug_info,
    flatten_fun as flatten_fun,
    flatten_fun_nokwargs as flatten_fun_nokwargs,
)
from jax._src.core import (
    Literal as Literal,
    Primitive as Primitive,
    Trace as Trace,
    Tracer as Tracer,
    call_p as call_p,
    get_aval as get_aval,
    typeof as typeof,
)
from jax._src.dtypes import (
    dtype as dtype,
    float0 as float0,
)
from jax._src.state.types import AbstractRef as AbstractRef
from jax._src.tree_util import (
    PyTreeDef as PyTreeDef,
    register_pytree_node as register_pytree_node,
    tree_flatten as tree_flatten,
    tree_unflatten as tree_unflatten,
)
from jax._src.util import (
    as_hashable_function as as_hashable_function,
    foreach as foreach,
    partition_list as partition_list,
    safe_map as safe_map,
    safe_zip as safe_zip,
    split_list as split_list,
    subs_list2 as subs_list2,
    unzip2 as unzip2,
    weakref_lru_cache as weakref_lru_cache,
    wrap_name as wrap_name,
)

type Array = Any
type Ref = Any
zip = safe_zip
map = safe_map

def identity(x): ...
def jvp(
    fun: lu.WrappedFun,
    has_aux: bool = False,
    instantiate: bool = True,
    transform_stack: bool = True,
) -> Any: ...
@lu.transformation2
def jvpfun(f: Callable, instantiate, transform_stack, primals, tangents): ...
@lu.transformation_with_aux2
def linearize_subtrace(
    _f: Callable,
    _store: lu.Store,
    _is_vjp: bool,
    _tag: core.TraceTag,
    nzs_in: Sequence[bool],
    debug_info: core.DebugInfo,
    *primals,
    **params,
): ...
@lu.transformation2
def jvp_subtrace(f: Callable, tag: core.TraceTag, primals, tangents): ...
@lu.transformation_with_aux2
def jvp_subtrace_aux(f, store, tag, primals, tangents): ...
def linearize_jaxpr(
    jaxpr: core.ClosedJaxpr,
    nonzeros: Sequence[bool],
    instantiate: bool | Sequence[bool] = False,
    allow_fwds: bool | Sequence[bool] = True,
    *,
    is_vjp: bool,
) -> tuple[
    core.ClosedJaxpr,
    int,
    Sequence[bool],
    Sequence[int | None],
    core.ClosedJaxpr,
]: ...
def direct_linearize(traceable, primals, *, has_aux, is_vjp): ...
def linearize(
    traceable: lu.WrappedFun,
    *primals,
    has_aux: bool = False,
    is_vjp: bool = False,
): ...

class UndefinedPrimal:
    aval: Incomplete
    def __init__(self, aval) -> None: ...

def is_undefined_primal(x): ...
def get_primitive_transpose(p): ...
def backward_pass3(
    jaxpr: core.Jaxpr,
    transform_stack: bool,
    consts: Sequence[Array],
    primals_in: Sequence[Array | Ref | GradAccum],
    cotangents_in: Sequence[Array],
) -> None: ...

class GradAccum:
    aval: core.AbstractValue
    def accum(self, x) -> None: ...
    def freeze(self) -> Array | Zero: ...

class RefAccum(GradAccum):
    aval: core.AbstractValue
    ref: AbstractRef | None
    def __init__(self, aval, ref=None) -> None: ...
    def accum(self, x) -> None: ...
    def freeze(self): ...
    def inst(self): ...

class ValAccum(GradAccum):
    aval: core.AbstractValue
    val: Array | Zero
    def __init__(self, aval, val=None) -> None: ...
    def accum(self, x) -> None: ...
    def freeze(self): ...

def ct_check(primal, ct) -> None: ...

class NullAccum(GradAccum):
    def __init__(self) -> None: ...
    def accum(self, x) -> None: ...
    def freeze(self) -> None: ...

fancy_transposes: dict[core.Primitive, Callable]

def project_accums(args): ...
def unproject_accums(specs, result): ...
def accum_typeof(x): ...
def backward_pass(jaxpr, transform_stack: bool, consts, primals_in, cts_in): ...
def closed_backward_pass(
    jaxpr: core.ClosedJaxpr,
    transform_stack,
    primals_in,
    cotangents_in,
): ...
@lu.transformation_with_aux2
def nonzero_tangent_outputs(f, store, *args, **kwargs): ...

class JVPTrace(Trace):
    tag: Incomplete
    parent_trace: Incomplete
    requires_low: bool
    def __init__(self, parent_trace, tag) -> None: ...
    def to_primal_tangent_pair(self, val): ...
    def process_primitive(self, primitive, tracers, params): ...
    def cur_qdd(self, x): ...
    def process_call(self, call_primitive, f, tracers, params): ...
    process_map = process_call
    def process_custom_jvp_call(self, prim, fun, f_jvp, tracers, *, symbolic_zeros): ...
    def process_custom_vjp_call(
        self,
        prim,
        fun,
        fwd,
        bwd,
        tracers,
        out_trees,
        symbolic_zeros,
    ): ...
    def process_custom_transpose(self, prim, call, tracers, **params): ...

def maybe_jvp_tracer(trace, primal, tangent): ...

class JVPTracer(Tracer):
    primal: Incomplete
    tangent: Incomplete
    def __init__(self, trace, primal, tangent) -> None: ...
    @property
    def aval(self): ...
    def cur_qdd(self): ...
    def full_lower(self): ...
    def to_concrete_value(self): ...
    def get_referent(self): ...
    def type_state(self): ...

call_param_updaters: dict[core.Primitive, Callable]
call_linearize_param_updaters: dict[core.Primitive, Callable]
call_transpose_param_updaters: dict[core.Primitive, Callable]

class LinearizeTrace(Trace):
    parent_trace: Incomplete
    tangent_trace: Incomplete
    is_vjp: Incomplete
    tag: Incomplete
    requires_low: bool
    def __init__(self, parent_trace, tangent_trace, is_vjp, tag) -> None: ...
    def to_primal_tangent_pair(self, val): ...
    def process_primitive(self, primitive, args, params): ...
    def cur_qdd(self, x): ...
    def process_custom_jvp_call(
        self,
        prim,
        fun: lu.WrappedFun,
        f_jvp: lu.WrappedFun,
        tracers,
        *,
        symbolic_zeros: bool,
    ): ...
    def process_custom_vjp_call(
        self,
        prim,
        fun,
        fwd,
        bwd: lu.WrappedFun,
        tracers,
        out_trees: Callable[[], tuple[PyTreeDef, PyTreeDef, list[int | None]]],
        symbolic_zeros: bool,
    ): ...
    def process_call(self, call_primitive, f: lu.WrappedFun, tracers, params): ...
    process_map = process_call

def maybe_linearize_tracer(trace, primal, is_nonzero, tangent): ...
def fallback_linearize_rule(
    _prim: core.Primitive,
    _is_vjp,
    _nonzeros: Sequence[bool],
    *primals,
    **params,
): ...
def linearize_from_jvp(
    jvp: lu.WrappedFun,
    multiple_results: bool,
    nonzeros: Sequence[bool],
    user_facing_symbolic_zeros: bool,
    instantiate_input_zeros: bool,
    primals,
    params,
): ...

class LinearizeTracer(Tracer):
    primal: Incomplete
    tangent: Incomplete
    def __init__(self, trace, primal, tangent) -> None: ...
    @property
    def aval(self): ...
    def full_lower(self): ...
    def to_concrete_value(self): ...
    def get_referent(self): ...
    def cur_qdd(self): ...

primitive_jvps: dict[core.Primitive, Callable]
primitive_transposes: dict[core.Primitive, Callable]
primitive_linearizations: dict[core.Primitive, Callable]

def deflinear(primitive, transpose_rule) -> None: ...
def linear_jvp(primitive, primals, tangents, **params): ...
def linear_transpose(transpose_rule, cotangent, *args, **kwargs): ...
def deflinear2(primitive, transpose_rule) -> None: ...
def linear_transpose2(transpose_rule, cotangent, *args, **kwargs): ...
def defjvp(primitive, *jvprules) -> None: ...
def standard_jvp(jvprules, primitive, primals, tangents, **params): ...
def defjvp2(primitive, *jvprules) -> None: ...
def standard_jvp2(jvprules, primitive, primals, tangents, **params): ...
def add_tangents(x, y): ...
def defbilinear(prim, lhs_rule, rhs_rule): ...
def bilinear_transpose(lhs_rule, rhs_rule, cotangent, x, y, **kwargs): ...
def defjvp_zero(primitive) -> None: ...
def zero_jvp(primitive, primals, tangents, **params): ...
def instantiate_zeros(tangent): ...
@lu.transformation_with_aux2
def traceable(f, store, in_tree, *primals_and_tangents): ...
def call_transpose_fancy(primitive, cts, *args, call_jaxpr, **params): ...
@lu.transformation_with_aux2
def nonzero_outputs(f, store, *args, **kwargs): ...
def map_transpose(
    primitive: core.Primitive,
    params,
    call_jaxpr: core.Jaxpr,
    args,
    ct,
    _,
): ...
def jvp_jaxpr(
    jaxpr: core.ClosedJaxpr,
    nonzeros: Sequence[bool],
    instantiate: bool | Sequence[bool],
) -> tuple[core.ClosedJaxpr, list[bool]]: ...
@lu.transformation_with_aux2
def f_jvp_traceable(f, store, nonzeros, *primals_and_nztangents): ...
def rearrange_binders(
    jaxpr: core.ClosedJaxpr,
    primals_in,
    tangents_in,
    primals_out,
    tangents_out,
): ...

custom_lin_p: core.Primitive

def raise_custom_vjp_error_on_jvp(*_, **__) -> None: ...

class CustomJVPException(Exception):
    def __init__(self) -> None: ...

class CustomVJPException(Exception):
    def __init__(self) -> None: ...

reducing_transposes: dict[core.Primitive, Callable]

def call_transpose(primitive, params, call_jaxpr: core.Jaxpr, args, ct, _): ...
