from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

import dataclasses

from _typeshed import Incomplete
from jax import (
    lax as lax,
    tree_util as tree_util,
)
from jax._src import (
    ad_checkpoint as ad_checkpoint,
    ad_util as ad_util,
    api_util as api_util,
    config as config,
    core as jax_core,
    custom_derivatives as custom_derivatives,
    debugging as debugging,
    literals as literals,
    pjit as pjit,
    source_info_util as source_info_util,
    state as state,
    util as util,
)
from jax._src.interpreters import mlir as mlir
from jax._src.lib.mlir import ir as ir
from jax._src.pallas import (
    core as pallas_core,
    primitives as primitives,
)
from jax._src.state import indexing as indexing
from jax._src.util import (
    foreach as foreach,
    split_list as split_list,
)

import jax

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
NDIndexer: Incomplete
GridMapping: Incomplete
BlockMapping: Incomplete
Blocked: Incomplete

@dataclasses.dataclass
class ModuleContext:
    name: str
    grid_mapping: GridMapping
    program_ids: Sequence[ir.Value]
    traceback_caches: mlir.TracebackCaches = ...
    platform: str

@dataclasses.dataclass
class BlockInfo:
    full_shape_dtype: jax_core.ShapedArray
    start_indices: Sequence[Any]
    start_indices_alignment: Sequence[int]
    block_shape: tuple[int | pallas_core.Squeezed, ...]

@dataclasses.dataclass
class LoweringRuleContext:
    context: ModuleContext
    avals_in: Sequence[jax_core.ShapedArray]
    avals_out: Sequence[jax_core.ShapedArray]
    block_infos: Sequence[BlockInfo | None]
    replace = dataclasses.replace

@dataclasses.dataclass
class LoweringResult:
    module: ir.Module
    grid: tuple[int, ...]

class LoweringError(Exception): ...

triton_lowering_rules: Incomplete

def register_lowering(primitive: jax_core.Primitive) -> Callable[[_T], _T]: ...
def lower_jaxpr_to_triton_module(
    jaxpr: jax_core.Jaxpr,
    grid_mapping: GridMapping,
    platform: str,
) -> LoweringResult: ...
def lower_jaxpr_to_triton_ir(
    ctx: ModuleContext,
    jaxpr: jax_core.Jaxpr,
    block_infos: Sequence[BlockInfo | None] | None,
    *args,
) -> Sequence[Any]: ...
def lower_fun(
    fun: Callable[..., Any],
    *,
    multiple_results: bool,
) -> Callable[..., Any]: ...

@dataclasses.dataclass(frozen=True)
class _Extern:
    arg_types: Sequence[jax.typing.DTypeLike]
    symbol: str
    result_type: str
    def matches(self, avals: Sequence[jax_core.ShapedArray]) -> bool: ...
    def lower(self, ctx: LoweringRuleContext, *args: Sequence[ir.Value]): ...

@dataclasses.dataclass(frozen=True)
class _Fallback:
    arg_classes: Sequence[jax.typing.DTypeLike]
    op: Callable[..., ir.Value]
    def matches(self, avals: Sequence[jax_core.ShapedArray]) -> bool: ...
    def lower(self, ctx: LoweringRuleContext, *args: Sequence[ir.Value]): ...

abs_dispatch_table: Incomplete
ceil_dispatch_table: Incomplete
floor_dispatch_table: Incomplete
exp_dispatch_table: Incomplete
exp2_dispatch_table: Incomplete
expm1_dispatch_table: Incomplete
log_dispatch_table: Incomplete
log1p_dispatch_table: Incomplete
sqrt_dispatch_table: Incomplete
pow_dispatch_table: Incomplete
cbrt_dispatch_table: Incomplete
rsqrt_dispatch_table: Incomplete
sin_dispatch_table: Incomplete
cos_dispatch_table: Incomplete
tan_dispatch_table: Incomplete
asin_dispatch_table: Incomplete
acos_dispatch_table: Incomplete
atan_dispatch_table: Incomplete
atan2_dispatch_table: Incomplete
sinh_dispatch_table: Incomplete
cosh_dispatch_table: Incomplete
tanh_dispatch_table: Incomplete
asinh_dispatch_table: Incomplete
acosh_dispatch_table: Incomplete
atanh_dispatch_table: Incomplete
population_count_dispatch_table: Incomplete
clz_dispatch_table: Incomplete
nextafter_dispatch_table: Incomplete

def signless_rule(ctx: LoweringRuleContext, x, y, fn=...): ...
def signed_rule(ctx: LoweringRuleContext, x, y, fn=...): ...
def debug_print_lowering_rule(
    ctx: LoweringRuleContext,
    *args: ir.Value,
    fmt: str,
    ordered,
    partitioned,
    in_tree,
    static_args,
    np_printoptions,
    has_placeholders,
    logging_record,
): ...
def select_n_lowering_rule(ctx: LoweringRuleContext, pred, x, y): ...
def get_join_type(old_type: ir.RankedTensorType): ...
