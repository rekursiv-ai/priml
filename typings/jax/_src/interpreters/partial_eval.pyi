from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple
from weakref import ReferenceType, WeakValueDictionary, ref

import contextlib

from _typeshed import Incomplete
from jax._src import (
    ad_util as ad_util,
    api_util as api_util,
    config as config,
    core as core,
    dtypes as dtypes,
    effects as effects,
    linear_util as lu,
    profiler as profiler,
    source_info_util as source_info_util,
    tree_util as tree_util,
    xla_metadata_lib as xla_metadata_lib,
)
from jax._src.core import (
    AbstractValue as AbstractValue,
    Atom as Atom,
    ClosedJaxpr as ClosedJaxpr,
    DropVar as DropVar,
    Jaxpr as Jaxpr,
    JaxprEqn as JaxprEqn,
    JaxprEqnContext as JaxprEqnContext,
    Literal as Literal,
    Primitive as Primitive,
    Trace as Trace,
    Tracer as Tracer,
    TraceTag as TraceTag,
    Var as Var,
    get_aval as get_aval,
    get_referent as get_referent,
    mapped_aval as mapped_aval,
    new_jaxpr_eqn as new_jaxpr_eqn,
    typeof as typeof,
    unmapped_aval as unmapped_aval,
)
from jax._src.lib import jaxlib_extension_version as jaxlib_extension_version
from jax._src.source_info_util import SourceInfo as SourceInfo
from jax._src.state.types import (
    AbstractRef as AbstractRef,
    ReadEffect as ReadEffect,
)
from jax._src.tree_util import (
    FlatTree as FlatTree,
    PyTreeDef as PyTreeDef,
    treedef_tuple as treedef_tuple,
)
from jax._src.util import (
    HashableFunction as HashableFunction,
    OrderedSet as OrderedSet,
    as_hashable_function as as_hashable_function,
    foreach as foreach,
    merge_lists as merge_lists,
    multi_weakref_lru_cache as multi_weakref_lru_cache,
    partition_list as partition_list,
    safe_map as safe_map,
    safe_zip as safe_zip,
    split_list as split_list,
    subs_list as subs_list,
    test_event as test_event,
    toposort as toposort,
    unzip2 as unzip2,
    weakref_lru_cache as weakref_lru_cache,
)

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete

def identity(x): ...

TracerId = int
AvalId = int
ConstId = int
type AttrKind = Any
type PyTree = Any
logger: Incomplete
TracebackScope: Incomplete
TracebackScope = contextlib.nullcontext

class PartialVal(tuple):
    def __new__(cls, xs: tuple[AbstractValue | None, core.Value]): ...
    @classmethod
    def known(cls, const: core.Value) -> PartialVal: ...
    @classmethod
    def unknown(cls, aval: AbstractValue) -> PartialVal: ...
    def is_known(self) -> bool: ...
    def get_known(self) -> core.Value | None: ...
    def get_aval(self) -> AbstractValue: ...

@dataclass(frozen=True)
class EffectHandle:
    parents: list[Tracer]
    recipe: JaxprEqnRecipe

class JaxprTrace(Trace["JaxprTracer"]):
    name_stack: Incomplete
    tag: Incomplete
    parent_trace: Incomplete
    requires_low: bool
    effect_handles: list[EffectHandle]
    counter: Incomplete
    def __init__(
        self,
        parent_trace: Trace,
        name_stack: source_info_util.NameStack,
        tag: TraceTag,
    ) -> None: ...
    def to_jaxpr_tracer(self, x): ...
    def new_const(self, val) -> JaxprTracer: ...
    def new_instantiated_literal(self, val) -> JaxprTracer: ...
    def new_instantiated_const(self, val) -> JaxprTracer: ...
    def new_arg(self, pval: PartialVal) -> JaxprTracer: ...
    def instantiate_const(self, tracer: JaxprTracer) -> JaxprTracer: ...
    def cur_qdd(self, x): ...
    def process_primitive(self, primitive, tracers, params): ...
    def default_process_primitive(self, primitive, tracers, params): ...
    def process_call(self, primitive, f: lu.WrappedFun, tracers, params): ...
    def process_map(self, primitive, f: lu.WrappedFun, tracers, params): ...
    def process_custom_jvp_call(self, prim, fun, jvp, tracers, symbolic_zeros): ...
    def process_custom_transpose(self, prim, call, tracers, **params): ...
    def process_custom_vjp_call(
        self,
        prim,
        f,
        fwd,
        bwd,
        tracers,
        out_trees,
        symbolic_zeros,
    ): ...

def partition_pvals(
    pvals: list[PartialVal],
) -> tuple[list[bool], list[AbstractValue], list[Any]]: ...
@lu.transformation_with_aux2
def partial_eval_wrapper_nounits(
    f: Callable,
    store: lu.Store,
    in_knowns: Sequence[bool],
    in_avals: Sequence[AbstractValue],
    *in_consts: Any,
): ...
@lu.transformation_with_aux2
def partial_eval_wrapper_nounits2(
    f: Callable,
    store: lu.Store,
    in_knowns: Sequence[bool],
    in_avals: Sequence[AbstractValue],
    *in_consts: Any,
): ...

custom_partial_eval_rules: dict[Primitive, Callable]
call_partial_eval_rules: dict[Primitive, Callable]
call_param_updaters: dict[Primitive, Callable]

def abstract_eval_fun(fun: Callable, *avals, debug_info: core.DebugInfo, **params): ...

JaxprTracerRecipe: Incomplete

class JaxprTracer(Tracer):
    pval: Incomplete
    recipe: Incomplete
    def __init__(
        self,
        trace: JaxprTrace,
        pval: PartialVal,
        recipe: JaxprTracerRecipe | None,
    ) -> None: ...
    @property
    def aval(self) -> AbstractValue: ...
    @property
    def parents(self) -> Sequence[JaxprTracer]: ...
    def full_lower(self): ...
    def is_known(self): ...
    def get_referent(self): ...

@profiler.annotate_function
def trace_to_jaxpr_nounits(
    fun: lu.WrappedFun,
    pvals: Sequence[PartialVal],
    instantiate: bool | Sequence[bool] = False,
) -> tuple[Jaxpr, list[PartialVal], list[core.Value]]: ...
@lu.transformation2
def trace_to_subjaxpr_nounits(
    f: Callable,
    trace: JaxprTrace,
    instantiate: Sequence[bool] | bool,
    debug_info: core.DebugInfo,
    in_pvals: Sequence[PartialVal],
): ...
@lu.transformation2
def trace_to_subjaxpr_nounits2(
    f: Callable,
    tag: TraceTag,
    debug_info: core.DebugInfo,
    instantiate: bool | Sequence[bool],
    in_pvals: Sequence[PartialVal],
): ...
@lu.transformation2
def trace_to_subjaxpr_nounits_fwd(
    f: Callable,
    tag: TraceTag,
    debug_info: core.DebugInfo,
    instantiate: bool | Sequence[bool],
    in_pvals: Sequence[PartialVal],
): ...
@lu.transformation2
def trace_to_subjaxpr_nounits_fwd2(
    f: Callable,
    tag: TraceTag,
    debug_info: core.DebugInfo,
    instantiate: bool | Sequence[bool],
    in_pvals: Sequence[PartialVal],
): ...

class FreeVar(NamedTuple):
    val: Incomplete

class ConstVar(NamedTuple):
    val: Incomplete

class LambdaBinding(NamedTuple): ...

class JaxprEqnRecipe(NamedTuple):
    eqn_id: Any
    in_tracers: Sequence[JaxprTracer]
    out_tracer_refs: Sequence[ref[JaxprTracer]]
    out_avals: Sequence[core.AbstractValue]
    primitive: Primitive
    params: dict[str, Any]
    effects: core.Effects
    source_info: source_info_util.SourceInfo
    ctx: JaxprEqnContext

def new_eqn_recipe(
    trace: JaxprTrace,
    in_tracers: Sequence[JaxprTracer],
    out_tracers: Sequence[JaxprTracer],
    primitive: Primitive,
    params: dict[str, Any],
    effects: core.Effects,
    source_info: source_info_util.SourceInfo,
    ctx: JaxprEqnContext | None = None,
) -> JaxprEqnRecipe: ...
def tracers_to_jaxpr(
    in_tracers: Sequence[JaxprTracer],
    out_tracers: Sequence[JaxprTracer],
    effect_handles: Sequence[Any],
    debug_info: core.DebugInfo,
) -> tuple[Jaxpr, tuple[Any, ...], tuple[Any, ...]]: ...
@weakref_lru_cache
def move_envvars(jaxpr: Jaxpr, which: tuple[bool, ...]) -> Jaxpr: ...
@weakref_lru_cache
def separate_consts(jaxpr: ClosedJaxpr) -> tuple[ClosedJaxpr, list[Any]]: ...
@weakref_lru_cache
def convert_constvars_jaxpr(jaxpr: Jaxpr) -> Jaxpr: ...
@weakref_lru_cache
def convert_invars_to_constvars(jaxpr: Jaxpr, n: int) -> Jaxpr: ...
def partial_eval_jaxpr_nounits(
    jaxpr: ClosedJaxpr,
    unknowns: Sequence[bool],
    instantiate: bool | Sequence[bool],
) -> tuple[ClosedJaxpr, ClosedJaxpr, list[bool], list[AbstractValue]]: ...
def partial_eval_jaxpr_nounits_fwd(
    jaxpr: ClosedJaxpr,
    unknowns: Sequence[bool],
    instantiate: bool | Sequence[bool],
    fwd: bool | Sequence[bool] = True,
) -> tuple[
    ClosedJaxpr,
    ClosedJaxpr,
    list[bool],
    list[AbstractValue],
    list[int | None],
]: ...
def partial_eval_jaxpr_custom(
    jaxpr: Jaxpr,
    in_unknowns: Sequence[bool],
    in_inst: bool | Sequence[bool],
    ensure_out_unknowns: bool | Sequence[bool],
    ensure_out_inst: bool | Sequence[bool],
    saveable: Callable[..., RematCases_],
) -> tuple[Jaxpr, Jaxpr, list[bool], list[bool], int]: ...
def partial_eval_jaxpr_stateful(
    jaxpr: Jaxpr,
    in_unknowns: Sequence[bool],
    in_inst: bool | Sequence[bool],
    ensure_out_unknowns: bool | Sequence[bool],
    ensure_out_inst: bool | Sequence[bool],
    saveable: Callable[..., RematCases_] | None,
) -> tuple[Jaxpr, Jaxpr, list[bool], list[bool], int, int]: ...

everything_saveable: Incomplete
MemoryKind = str

class RecomputeType: ...

Recompute: Incomplete

class SaveableType: ...

Saveable: Incomplete

class Offloadable(NamedTuple):
    src: MemoryKind
    dst: MemoryKind

type RematCases = RecomputeType | SaveableType | Offloadable
type RematCases_ = RematCases | bool

def ensure_enum(case: bool | RematCases) -> RematCases: ...

type PartialEvalCustomResult = tuple[
    JaxprEqn | None,
    JaxprEqn | None,
    Sequence[bool],
    Sequence[bool],
    list[Var],
]
type PartialEvalCustomRule = Callable[
    [Callable[..., RematCases_], Sequence[bool], Sequence[bool], JaxprEqn],
    PartialEvalCustomResult,
]
partial_eval_jaxpr_custom_rules: dict[Primitive, PartialEvalCustomRule]
type ParamsUpdater = Callable[
    [Sequence[bool], Sequence[bool], Sequence[bool], Sequence[bool], int, dict, dict],
    tuple[dict, dict],
]
type ResAvalUpdater = Callable[[dict[str, Any], AbstractValue], AbstractValue]

def call_partial_eval_custom_rule(
    jaxpr_param_name: str,
    params_updater: ParamsUpdater,
    saveable: Callable[..., RematCases_],
    unks_in: list[bool],
    inst_in: list[bool],
    eqn: JaxprEqn,
    *,
    res_aval: ResAvalUpdater = ...,
    ctx=...,
) -> tuple[JaxprEqn, JaxprEqn, Sequence[bool], Sequence[bool], list[Var]]: ...

type ParamsUpdater2 = Callable[
    [
        Sequence[bool],
        Sequence[bool],
        Sequence[bool],
        Sequence[bool],
        int,
        int,
        dict,
        dict,
    ],
    tuple[dict, dict],
]

def closed_call_partial_eval_custom_rule(
    jaxpr_param_name: str,
    params_updater: ParamsUpdater2,
    saveable: Callable[..., RematCases_],
    unks_in: list[bool],
    inst_in: list[bool],
    eqn: JaxprEqn,
    *,
    res_aval: ResAvalUpdater = ...,
) -> tuple[JaxprEqn, JaxprEqn, Sequence[bool], Sequence[bool], list[Var]]: ...
def prune_jaxpr_outputs(jaxpr: Jaxpr, used_outputs: Sequence[bool]) -> Jaxpr: ...
def prune_closed_jaxpr_outputs(
    jaxpr: ClosedJaxpr,
    used_outputs: Sequence[bool],
) -> ClosedJaxpr: ...
def dce_jaxpr(
    jaxpr: Jaxpr,
    used_outputs: Sequence[bool],
    instantiate: bool | Sequence[bool] = False,
) -> tuple[Jaxpr, list[bool]]: ...
def dce_jaxpr_consts(
    jaxpr: Jaxpr,
    used_outputs: Sequence[bool],
    instantiate: bool | Sequence[bool] = False,
) -> tuple[Jaxpr, list[bool], list[bool]]: ...

dce_rules: dict[Primitive, DCERule]
dceable_effects: Incomplete

def has_effects(eqn: JaxprEqn) -> bool: ...

type DCERule = Callable[[list[bool], JaxprEqn], tuple[list[bool], JaxprEqn | None]]

def dce_jaxpr_call_rule(
    used_outputs: list[bool],
    eqn: JaxprEqn,
) -> tuple[list[bool], JaxprEqn | None]: ...
def dce_jaxpr_closed_call_rule(
    used_outputs: list[bool],
    eqn: JaxprEqn,
) -> tuple[list[bool], JaxprEqn | None]: ...
@weakref_lru_cache
def close_jaxpr(jaxpr: Jaxpr) -> ClosedJaxpr: ...
def move_invars_right(jaxpr: ClosedJaxpr, to_move: Sequence[bool]): ...
def move_binders_to_front(
    closed_jaxpr: ClosedJaxpr,
    to_move: Sequence[bool],
) -> ClosedJaxpr: ...
def move_binders_to_back(
    closed_jaxpr: ClosedJaxpr,
    to_move: Sequence[bool],
) -> ClosedJaxpr: ...

class DynamicJaxprTracer(core.Tracer):
    aval: Incomplete
    val: Incomplete
    mutable_qdd: Incomplete
    parent: Incomplete
    def __init__(
        self,
        trace: DynamicJaxprTrace,
        aval: core.AbstractValue | core.AvalQDD,
        val: Atom,
        line_info: source_info_util.SourceInfo | None = None,
        parent: TracingEqn | None = None,
    ) -> None: ...
    def cur_qdd(self): ...
    @property
    def aval_mutable_qdd(self): ...
    def full_lower(self): ...
    def get_const(self): ...
    def get_referent(self): ...

def make_jaxpr_effects(constvars, invars, outvars, eqns) -> effects.Effects: ...

class Constants(NamedTuple):
    canonical: Any
    original: Any

class JaxprStackFrame:
    gensym: Callable[[AbstractValue], Var]
    constid_to_tracer: WeakValueDictionary[ConstId, DynamicJaxprTracer]
    constvar_to_val: dict[Var, Constants]
    tracing_eqns: list[ReferenceType[TracingEqn] | Callable[[], TracingEqn]]
    invars: list[Var]
    effects: core.Effects
    debug_info: core.DebugInfo
    is_high: bool
    mutable_qdds: list[tuple[Var, core.MutableQuasiDynamicData]]
    auto_dce: bool
    def __init__(self, debug_info: core.DebugInfo, auto_dce: bool) -> None: ...
    def add_eqn(self, eqn: TracingEqn): ...
    def get_eqns(self): ...
    def to_jaxpr(
        self,
        trace: DynamicJaxprTrace,
        out_tracers: Sequence[Tracer],
        debug_info: core.DebugInfo,
        source_info: SourceInfo,
    ) -> tuple[Jaxpr, list[Any]]: ...
    def newvar(self, aval): ...
    def find_progenitors(self, tracer): ...

type ConstFoldRule = Callable[
    [list[Any | None], Any, list[AbstractValue]],
    tuple[list[Any | None], JaxprEqn | None],
]
const_fold_rules: dict[Primitive, ConstFoldRule]
type ForwardingRule = Callable[[JaxprEqn], tuple[list[int | None], JaxprEqn | None]]
forwarding_rules: dict[Primitive, ForwardingRule]

@dataclass
class TracingEqn:
    in_tracers: list[DynamicJaxprTracer]
    outvars: list[Var]
    primitive: Primitive
    params: dict[str, Any]
    effects: core.Effects
    source_info: source_info_util.SourceInfo
    ctx: JaxprEqnContext
    def __init__(
        self,
        in_tracers,
        outvars,
        primitive,
        params,
        effects,
        source_info,
        ctx,
    ) -> None: ...
    @property
    def invars(self): ...

class DynamicJaxprTrace(core.Trace):
    requires_low: Incomplete
    frame: Incomplete
    parent_trace: Incomplete
    def __init__(
        self,
        debug_info: core.DebugInfo,
        parent_trace=None,
        lower: bool = False,
        auto_dce: bool = False,
    ) -> None: ...
    def invalidate(self) -> None: ...
    def to_jaxpr_tracer(self, x, source_info: SourceInfo): ...
    def var_to_tracer(self, var, source_info, parent=None): ...
    def new_arg(self, aval, source_info: SourceInfo): ...
    def make_eqn(
        self,
        in_tracers,
        out_avals,
        primitive,
        params,
        effects,
        source_info=None,
        ctx=None,
    ): ...
    def emit_eqn(
        self,
        in_tracers,
        out_avals,
        primitive,
        params,
        effects,
        source_info=None,
        ctx=None,
    ): ...
    def new_const(
        self,
        c,
        source_info: SourceInfo,
        aval: AbstractValue | None = None,
    ): ...
    pure = new_const
    lift = new_const
    def finalize_const(self, var, constid) -> None: ...
    def get_const(self, tracer) -> Any: ...
    def cur_qdd(self, x): ...
    def process_primitive(self, primitive, tracers, params): ...
    def default_process_primitive(
        self,
        primitive,
        tracers,
        params,
        source_info=None,
    ): ...
    def process_call(self, call_primitive, f: lu.WrappedFun, in_tracers, params): ...
    def process_map(self, map_primitive, f: lu.WrappedFun, tracers, params): ...
    def process_custom_jvp_call(
        self,
        prim,
        fun: lu.WrappedFun,
        jvp: lu.WrappedFun,
        tracers,
        symbolic_zeros: bool,
    ): ...
    def process_custom_vjp_call(
        self,
        prim: core.Primitive,
        fun: lu.WrappedFun,
        fwd: lu.WrappedFun,
        bwd: lu.WrappedFun,
        tracers,
        out_trees: Callable[[], tuple[PyTreeDef, PyTreeDef, list[int | None]]],
        symbolic_zeros: bool,
    ): ...
    def process_custom_transpose(
        self,
        prim: core.Primitive,
        call: lu.WrappedFun,
        tracers,
        *,
        transpose: lu.WrappedFun,
        out_types,
        lin_tree: PyTreeDef,
        res_tree: PyTreeDef,
        out_tree: PyTreeDef,
    ): ...
    def to_jaxpr(
        self,
        out_tracers: Sequence[Tracer],
        debug_info: core.DebugInfo,
        source_info: SourceInfo,
    ): ...

custom_staging_rules: dict[Primitive, Callable]
callsites_with_tracing_cache_miss: set[str]

def explain(keys, fun, in_avals, debug_info, *context): ...
def diff_tracing_cache_keys(new_key, old_key) -> tuple[int, int, str]: ...
def diff_ctx(new_ctx, old_ctx): ...
def diff_trees(new_tree, old_tree): ...
def diff_debug(new_dbg, old_dbg): ...
def diff_types(dbg, new_leaves, old_leaves): ...
def trace_to_jaxpr(
    fun: Callable,
    in_avals: FlatTree,
    debug_info: core.DebugInfo,
    *context_for_cache_key,
) -> tuple[ClosedJaxpr, FlatTree]: ...
@profiler.annotate_function
def trace_to_jaxpr_dynamic(
    fun: lu.WrappedFun,
    in_avals: Sequence[AbstractValue | core.AvalQDD],
    *,
    keep_inputs: list[bool] | None = None,
    lower: bool = False,
    auto_dce: bool = False,
) -> tuple[Jaxpr, list[AbstractValue], list[Any]]: ...

class TracerAsName:
    ref: Any
    def __init__(self, tracer) -> None: ...
    def __eq__(self, other): ...
    def __hash__(self): ...

type Const = Any
type Val = Any

def inline_jaxpr_into_trace(
    trace: DynamicJaxprTrace,
    src: SourceInfo,
    jaxpr: Jaxpr,
    consts: Sequence[Any],
    *arg_tracers: DynamicJaxprTracer,
) -> list[Any]: ...
def try_constant_folding(primitive, tracers, params, out_avals): ...
@weakref_lru_cache
def lower_jaxpr(hi_jaxpr: core.ClosedJaxpr): ...
def lower_traceable(jaxpr, *lo_args): ...
@weakref_lru_cache
def convert_const_himutables(jaxpr): ...
def num_himuts_out(jaxpr): ...
def apply_himut(jaxpr: Jaxpr | ClosedJaxpr, hi_args, out_mut): ...
def raise_lo_outs(hi_avals, lo_outs): ...
