from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, NamedTuple, Protocol

import dataclasses
import typing

from _typeshed import Incomplete
from jax._src import (
    ad_util as ad_util,
    api_util as api_util,
    config as config,
    core as core,
    dtypes as dtypes,
    effects as effects_lib,
    frozen_dict as frozen_dict,
    hashable_array as hashable_array,
    jaxpr_util as jaxpr_util,
    literals as literals,
    path as path,
    sharding_impls as sharding_impls,
    source_info_util as source_info_util,
    util as util,
)
from jax._src.layout import (
    AutoLayout as AutoLayout,
    Layout as Layout,
)
from jax._src.lib import (
    jax_mlir_ext as jax_mlir_ext,
    jaxlib_extension_version as jaxlib_extension_version,
    xla_client as xc,
)
from jax._src.lib.mlir import (
    dialects as dialects,
    ir as ir,
    passmanager as passmanager,
)
from jax._src.lib.mlir.dialects import (
    func as func_dialect,
    hlo as hlo,
)
from jax._src.mesh import AxisType as AxisType
from jax._src.partition_spec import PartitionSpec as PartitionSpec
from jax._src.sharding import Sharding as JSharding
from jax._src.sharding_impls import (
    AUTO as AUTO,
    NamedSharding as NamedSharding,
    SdyArray as SdyArray,
    SdyArrayList as SdyArrayList,
    modify_sdy_sharding_wrt_axis_types as modify_sdy_sharding_wrt_axis_types,
)
from jax._src.state.types import AbstractRef as AbstractRef
from jax._src.typing import ArrayLike as ArrayLike
from jax._src.util import foreach as foreach

import numpy as np

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
T = typing.TypeVar("T")
type Value = Any
MYPY: bool
IrValues: Incomplete

def dense_int_elements(xs) -> ir.DenseIntElementsAttr: ...

dense_int_array: Incomplete

def i32_attr(i): ...
def i64_attr(i): ...
def shape_tensor(sizes: Sequence[int | ir.RankedTensorType]) -> ir.RankedTensorType: ...
def delegate_lowering(ctx, lowering_fun, *args, **ctx_override_kwargs): ...

IrTypes: Incomplete

def dtype_to_ir_type(dtype: core.bint | np.dtype | np.generic) -> ir.Type: ...

ir_type_handlers: dict[type[core.AbstractValue], Callable[[Any], IrTypes]]

def aval_to_ir_type(aval: core.AbstractValue) -> IrTypes: ...
def aval_to_ir_types(aval: core.AbstractValue) -> tuple[ir.Type, ...]: ...

class ConstantHandler(Protocol):
    def __call__(self, val: Any, aval: core.AbstractValue | None) -> IrValues: ...

def register_constant_handler(type_: type, handler_fun: ConstantHandler): ...
def get_constant_handler(type_: type) -> ConstantHandler: ...
def ir_constant(
    val: Any,
    *,
    const_lowering: dict[tuple[int, core.AbstractValue], IrValues] | None = None,
    aval: core.AbstractValue | None = None,
) -> IrValues: ...

AttributeHandler: Incomplete

def register_attribute_handler(type_: type[Any], handler_fun: AttributeHandler): ...
def get_attribute_handler(type_: type[Any]) -> AttributeHandler: ...
def ir_attribute(val: Any) -> ir.Attribute: ...
def get_canonical_source_file(file_name: str, caches: TracebackCaches) -> str: ...

class HasTracebackCaches(Protocol):
    @property
    def traceback_caches(self) -> TracebackCaches: ...

def source_info_to_location(
    ctx: HasTracebackCaches,
    primitive: core.Primitive | None,
    name_stack: source_info_util.NameStack,
    traceback: xc.Traceback | None,
) -> ir.Location: ...

upstream_dialects: Incomplete

def dump_module_to_file(module: ir.Module, stage_name: str) -> str | None: ...
def dump_module_message(module: ir.Module, stage_name: str) -> str: ...
def module_to_string(module: ir.Module, enable_debug_info=None) -> str: ...
def module_to_bytecode(module: ir.Module) -> bytes: ...

global_thread_pool: Incomplete

class JaxIrContext(ir.Context):
    def __init__(self, *args, **kwargs) -> None: ...

def make_ir_context() -> ir.Context: ...

AxisContext: Incomplete

class ShapePolyLoweringState:
    dim_vars: tuple[str, ...]
    uses_dim_vars: bool
    has_platform_index_argument: bool
    def __init__(
        self,
        dim_vars: tuple[str, ...],
        lowering_platforms: tuple[str, ...] | None,
    ) -> None: ...

@dataclasses.dataclass(frozen=True)
class LoweringParameters:
    override_lowering_rules: tuple[tuple[core.Primitive, LoweringRule]] | None = ...
    global_constant_computation: bool = ...
    for_export: bool = ...
    export_ignore_forward_compatibility: bool = ...
    hoist_constants_as_args: bool = ...

@dataclasses.dataclass
class TracebackCaches:
    traceback_to_location_cache: Any
    canonical_name_cache: dict[str, str]
    def __init__(self) -> None: ...

@dataclasses.dataclass(frozen=True)
class LoweringCacheKey:
    primitive: core.Primitive
    eqn_ctx: core.JaxprEqnContext
    avals_in: tuple[core.AbstractValue, ...]
    effects: effects_lib.Effects
    params: frozen_dict.FrozenDict[str, Any]
    platforms: tuple[str, ...]

@dataclasses.dataclass(frozen=True)
class LoweringCacheValue:
    func: func_dialect.FuncOp
    output_types: Sequence[IrTypes]
    const_args: Sequence[ArrayLike]
    const_arg_avals: Sequence[core.AbstractValue]
    inline: bool

@dataclasses.dataclass
class ModuleContext:
    context: ir.Context
    module: ir.Module
    ip: ir.InsertionPoint
    symbol_table: ir.SymbolTable
    platforms: Sequence[str]
    backend: xc.Client | None
    axis_context: AxisContext
    keepalives: list[Any]
    channel_iterator: Iterator[int]
    host_callbacks: list[Any]
    shape_poly_state: ShapePolyLoweringState
    all_default_mem_kind: bool
    lowering_cache: dict[LoweringCacheKey, LoweringCacheValue]
    cached_primitive_lowerings: dict[Any, func_dialect.FuncOp]
    traceback_caches: TracebackCaches
    lowering_parameters: LoweringParameters
    @property
    def axis_env(self) -> sharding_impls.AxisEnv: ...
    def __init__(
        self,
        *,
        platforms: Sequence[str],
        backend: xc.Client | None,
        axis_context: AxisContext,
        keepalives: list[Any],
        channel_iterator: Iterator[int],
        host_callbacks: list[Any],
        lowering_parameters: LoweringParameters,
        context: ir.Context | None = None,
        module: ir.Module | None = None,
        ip: ir.InsertionPoint | None = None,
        symbol_table: ir.SymbolTable | None = None,
        lowering_cache: dict[LoweringCacheKey, Any] | None = None,
        cached_primitive_lowerings: dict[Any, func_dialect.FuncOp] | None = None,
        traceback_caches: TracebackCaches | None = None,
        shape_poly_state=None,
        all_default_mem_kind: bool = True,
    ) -> None: ...
    def get_backend(self, optional: bool = False) -> xc.Client | None: ...
    def new_channel(self) -> int: ...
    def add_host_callback(self, host_callback: Any) -> None: ...
    def add_keepalive(self, keepalive: Any) -> None: ...
    def replace(self, **kw): ...

@dataclasses.dataclass
class LoweringRuleContext:
    module_context: ModuleContext
    name_stack: source_info_util.NameStack
    traceback: xc.Traceback | None
    primitive: core.Primitive | None
    avals_in: Sequence[core.AbstractValue]
    avals_out: Any
    tokens_in: TokenSet
    tokens_out: TokenSet | None
    const_lowering: dict[tuple[int, core.AbstractValue], IrValues]
    axis_size_env: dict[core.Var, ir.Value] | None = ...
    dim_var_values: Sequence[ir.Value] = ...
    jaxpr_eqn_ctx: core.JaxprEqnContext | None = ...
    platforms: Sequence[str] | None = ...
    def set_tokens_out(self, tokens_out: TokenSet): ...
    def replace(self, **kw): ...
    def is_forward_compat(self) -> bool: ...

type LoweringRule = Any

@dataclasses.dataclass(frozen=True)
class LoweringRuleEntry:
    rule: LoweringRule
    inline: bool

def register_lowering(
    prim: core.Primitive,
    rule: LoweringRule,
    platform: str | None = None,
    inline: bool = True,
    cacheable: bool = True,
) -> None: ...
def flatten_ir_values(xs: Iterable[IrValues]) -> list[ir.Value]: ...
def flatten_ir_types(xs: Iterable[IrTypes]) -> list[ir.Type]: ...
def len_ir_types(x: IrTypes) -> int: ...
def unflatten_ir_values_like_types(
    xs: Iterable[ir.Value],
    ys: Sequence[IrTypes],
) -> list[IrValues]: ...
def sanitize_name(name: str) -> str: ...
def sharded_aval(
    aval: core.AbstractValue,
    sharding: JSharding | AUTO | None,
) -> core.AbstractValue: ...
def eval_dynamic_shape(
    ctx: LoweringRuleContext,
    shape: core.Shape,
) -> tuple[int | Value, ...]: ...
def eval_dynamic_shape_as_vals(
    ctx: LoweringRuleContext,
    shape: core.Shape,
) -> tuple[Value, ...]: ...
def eval_dynamic_shape_as_ivals(
    ctx: LoweringRuleContext,
    shape: core.Shape,
) -> tuple[int | Value, ...]: ...
def eval_dynamic_shape_as_tensor(
    ctx: LoweringRuleContext,
    shape: core.Shape,
) -> Value: ...

class LoweringResult(NamedTuple):
    module: ir.Module
    keepalive: Any | None
    host_callbacks: list[Any]
    shape_poly_state: ShapePolyLoweringState

def add_manual_axes(axis_ctx: sharding_impls.SPMDAxisContext, sharding, ndim): ...
def contains_unconstrained(s): ...
def all_unconstrained(s, aval): ...

class UnconstrainedVariants(NamedTuple):
    contains_unconstrained: bool
    all_unconstrained: bool

def check_jaxpr_constants(closed_jaxpr: core.ClosedJaxpr): ...

COLLECTIVE_CHANNEL_ID: int

def lower_jaxpr_to_module(
    module_name: str,
    jaxpr: core.ClosedJaxpr,
    *,
    num_const_args: int,
    in_avals: Sequence[core.AbstractValue],
    ordered_effects: list[core.Effect],
    platforms: Sequence[str],
    backend: xc.Client | None,
    axis_context: AxisContext,
    donated_args: Sequence[bool],
    replicated_args: Sequence[bool] | None = None,
    arg_shardings: Sequence[JSharding | AUTO | None] | None = None,
    result_shardings: Sequence[JSharding | AUTO | None] | None = None,
    in_layouts: Sequence[Layout | AutoLayout | None] | None = None,
    out_layouts: Sequence[Layout | AutoLayout | None] | None = None,
    arg_names: Sequence[str] | None = None,
    result_names: Sequence[str] | None = None,
    num_replicas: int = 1,
    num_partitions: int = 1,
    all_default_mem_kind: bool = True,
    input_output_aliases: tuple[int | None, ...] | None = None,
    propagated_out_mem_kinds: tuple[str | None, ...] | None = None,
    lowering_parameters: LoweringParameters,
) -> LoweringResult: ...

Token: Incomplete
token_type: Incomplete
create_token: Incomplete

class TokenSet:
    def __init__(self, *args, **kwargs) -> None: ...
    def __len__(self) -> int: ...
    def get(self, effect: core.Effect) -> Token: ...
    @classmethod
    def create(cls, effects: Sequence[core.Effect]) -> TokenSet: ...
    def items(self) -> Sequence[tuple[core.Effect, Token]]: ...
    def effects(self) -> set[core.Effect]: ...
    def subset(self, effects: Sequence[core.Effect]) -> TokenSet: ...
    def update_tokens(self, tokens: TokenSet) -> TokenSet: ...

def lower_jaxpr_to_fun(
    ctx: ModuleContext,
    name: str,
    jaxpr: core.ClosedJaxpr,
    effects: Sequence[core.Effect],
    *,
    num_const_args: int,
    main_function: bool = False,
    replicated_args: Sequence[bool] | None = None,
    in_avals: Sequence[core.AbstractValue],
    arg_shardings: Sequence[JSharding | AUTO | None] | None = None,
    result_shardings: Sequence[JSharding | AUTO | None] | None = None,
    use_sharding_annotations: bool = True,
    input_output_aliases: Sequence[int | None] | None = None,
    xla_donated_args: Sequence[bool] | None = None,
    arg_names: Sequence[str | None] | None = None,
    result_names: Sequence[str] | None = None,
    arg_memory_kinds: Sequence[str | None] | None = None,
    result_memory_kinds: Sequence[str | None] | None = None,
    arg_layouts: Sequence[Layout | AutoLayout | None] | None = None,
    result_layouts: Sequence[Layout | AutoLayout | None] | None = None,
    propagated_out_mem_kinds: tuple[str | None, ...] | None = None,
) -> func_dialect.FuncOp: ...
def wrap_with_memory_kind(
    x: ir.Value,
    memory_kind: str,
    aval_out: core.AbstractValue,
) -> ir.Value: ...
def replicate_trailing_dims(ctx, val: ir.Value, aval) -> ir.Value: ...
def jaxpr_subcomp(
    ctx: ModuleContext,
    jaxpr: core.Jaxpr,
    name_stack: source_info_util.NameStack,
    tokens: TokenSet,
    consts_for_constvars: Sequence[IrValues],
    *args: IrValues,
    dim_var_values: Sequence[ir.Value],
    const_lowering: dict[tuple[int, core.AbstractValue], IrValues],
    outer_traceback: xc.Traceback | None,
) -> tuple[Sequence[IrValues], TokenSet]: ...
def lower_per_platform(
    ctx: LoweringRuleContext,
    description: str,
    platform_rules: dict[str, LoweringRule],
    default_rule: LoweringRule | None,
    effects: effects_lib.Effects,
    *rule_args: ir.Value | tuple[ir.Value, ...],
    **rule_kwargs,
) -> Sequence[ir.Value]: ...
def ir_consts(consts, avals: Sequence[core.AbstractValue]) -> list[IrValues]: ...
def lower_fun(fun: Callable, multiple_results: bool = True) -> Callable: ...
def check_backend_matches(
    inner_backend: str | None,
    lowering_platforms: Sequence[str],
): ...
def lower_called_computation(
    fn_name,
    call_jaxpr: core.ClosedJaxpr,
    ctx: ModuleContext,
    num_const_args: int,
    in_avals,
    out_avals,
    tokens_in,
    backend=None,
    arg_names=None,
    result_names=None,
): ...
def call_lowering(
    fn_name,
    call_jaxpr: core.ClosedJaxpr,
    backend,
    ctx: ModuleContext,
    in_avals,
    out_avals,
    tokens_in,
    *args,
    dim_var_values: Sequence[ir.Value],
    const_lowering: dict[tuple[int, core.AbstractValue], IrValues],
    arg_names=None,
    result_names=None,
    attributes: dict[str, Any] | None = None,
): ...
def core_call_lowering(
    ctx: LoweringRuleContext,
    *args,
    name,
    backend=None,
    call_jaxpr: core.ClosedJaxpr | core.Jaxpr,
): ...
def map_compute_type(c_type: str) -> str: ...
def wrap_compute_type_in_place(ctx: LoweringRuleContext, op: ir.Operation) -> None: ...
def wrap_xla_metadata_in_place(ctx: LoweringRuleContext, op: ir.Operation) -> None: ...
def broadcast_in_dim(
    ctx: LoweringRuleContext,
    op,
    aval_out: core.AbstractValue,
    *,
    broadcast_dimensions,
) -> ir.Value: ...
def multi_broadcast_in_dim(
    ctx: LoweringRuleContext,
    ops: Sequence[ir.Value],
    ops_avals: Sequence[core.AbstractValue],
    out_shape: core.Shape,
    out_sharding,
) -> Sequence[ir.Value]: ...
def reshape(ctx: LoweringRuleContext, op, aval_out: core.AbstractValue) -> ir.Value: ...
def slice_op(
    ctx: LoweringRuleContext,
    x,
    aval_out,
    *,
    start_indices,
    limit_indices,
    strides,
) -> ir.Value: ...
def dynamic_slice(
    ctx: LoweringRuleContext,
    aval_out,
    x,
    *,
    start_indices,
) -> ir.Value: ...
def dynamic_update_slice(
    ctx: LoweringRuleContext,
    aval_out,
    x,
    update,
    *,
    start_indices,
) -> ir.Value: ...
def pad(
    ctx: LoweringRuleContext,
    aval_out,
    x,
    padding_value,
    padding_low,
    padding_high,
    padding_interior,
) -> ir.Value: ...
def iota(ctx: LoweringRuleContext, aval_out, *, dimension: int): ...
def full_like_aval(
    ctx: LoweringRuleContext,
    value,
    aval: core.ShapedArray,
) -> ir.Value: ...
def add_jaxvals_lowering(ctx, x, y): ...
def compare_hlo(x, y, direction: str, comparison_type: str | None = None): ...

min_hlo: Incomplete
max_hlo: Incomplete

def convert_hlo(ctx: LoweringRuleContext, x, aval_in, aval_out): ...

wrap_with_sharding_op: Incomplete
wrap_with_full_to_shard_op: Incomplete
wrap_with_shard_to_full_op: Incomplete

def lower_with_sharding_in_types(ctx, op, aval, sharding_proto=None): ...
def set_sharding(op, sharding: xc.OpSharding | SdyArray | SdyArrayList): ...
def get_sharding_attr(
    sharding: xc.OpSharding | SdyArray | SdyArrayList,
) -> ir.Attribute: ...
def wrap_with_layout_op(
    ctx: LoweringRuleContext,
    x: ir.Value,
    aval_out: core.AbstractValue,
    layout: Layout,
    aval_in: core.AbstractValue,
): ...
def merge_mlir_modules(
    dst_module: ir.Module,
    sym_name: str,
    src_module: ir.Module,
    dst_symtab: ir.SymbolTable | None = None,
) -> str: ...

DEVICE_TO_DEVICE_TYPE: int
SEND_TO_HOST_TYPE: int
RECV_FROM_HOST_TYPE: int

def build_mlir_module_helper(
    closed_jaxpr: core.ClosedJaxpr,
    *,
    name: str,
    platforms: Sequence[str],
    backend: xc.Client | None,
    axis_context: AxisContext,
) -> ir.Module: ...
def custom_call(
    call_target_name: str,
    *,
    result_types: Sequence[ir.Type],
    operands: Sequence[ir.Value],
    backend_config: str | bytes | dict[str, ir.Attribute] = "",
    has_side_effect: bool = False,
    result_shapes: Sequence[ir.Value] | None = None,
    called_computations: Sequence[str] = (),
    api_version: int = 2,
    operand_output_aliases: dict[int, int] | None = None,
    operand_layouts: Sequence[Sequence[int]] | None = None,
    result_layouts: Sequence[Sequence[int]] | None = None,
    extra_attributes: dict[str, ir.Attribute] | None = None,
) -> ir.Operation: ...
def reduce_window(
    ctx: LoweringRuleContext,
    *,
    reducer_name: str,
    reducer_body: Callable[[ir.Block], Sequence[ir.Value]],
    operands: Sequence[ir.Value],
    init_values: Sequence[ir.Value],
    init_values_avals: Sequence[core.AbstractValue],
    out_avals: Sequence[core.AbstractValue],
    window_dimensions,
    window_strides,
    padding,
    base_dilation,
    window_dilation,
): ...
def refine_polymorphic_shapes(module: ir.Module) -> ir.Module: ...
