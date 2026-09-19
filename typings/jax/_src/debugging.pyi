from collections.abc import Callable, Generator, Sequence
from typing import Any

import string

from _typeshed import Incomplete
from jax._src import (
    api as api,
    config as config,
    core as core,
    dispatch as dispatch,
    effects as effects,
    lax as lax,
    shard_map as shard_map,
    sharding_impls as sharding_impls,
    source_info_util as source_info_util,
    tree_util as tree_util,
    util as util,
    xla_bridge as xla_bridge,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import hlo as hlo
from jax._src.sharding import Sharding as Sharding
from jax._src.sharding_impls import (
    NamedSharding as NamedSharding,
    parse_flatten_op_sharding as parse_flatten_op_sharding,
)

logger: Incomplete

class DebugEffect(effects.Effect): ...

debug_effect: Incomplete

class OrderedDebugEffect(effects.Effect): ...

ordered_debug_effect: Incomplete
debug_callback_p: Incomplete
map: Incomplete
unsafe_map: Incomplete

@debug_callback_p.def_impl
def debug_callback_impl(
    *args,
    callback: Callable[..., Any],
    effect: DebugEffect,
    partitioned: bool,
): ...
@debug_callback_p.def_effectful_abstract_eval
def debug_callback_abstract_eval(
    *flat_avals,
    callback: Callable[..., Any],
    effect: DebugEffect,
    partitioned: bool,
): ...
def debug_batching_rule(args, dims, *, primitive, **params): ...
def debug_callback_jvp_rule(primals, tangents, **params): ...
def debug_callback_transpose_rule(
    _,
    *flat_args,
    callback: Callable[..., Any],
    effect: DebugEffect,
    partitioned,
): ...
def debug_callback_lowering(ctx, *args, effect, partitioned, callback, **params): ...
def merge_callback_args(in_tree, dyn_args, static_args): ...

debug_print_p: Incomplete

@debug_print_p.def_impl
def debug_print_impl(
    *args: Any,
    fmt: str,
    ordered,
    partitioned,
    in_tree,
    static_args,
    np_printoptions,
    has_placeholders,
    logging_record,
): ...
@debug_print_p.def_effectful_abstract_eval
def debug_print_abstract_eval(*avals: Any, fmt: str, ordered, **kwargs): ...
def debug_print_jvp_rule(primals, tangents, **params): ...
def debug_print_transpose_rule(_, *args, **kwargs): ...
def debug_print_lowering_rule(
    ctx,
    *dyn_args,
    fmt,
    ordered,
    partitioned,
    in_tree,
    static_args,
    np_printoptions,
    has_placeholders,
    logging_record,
): ...
def debug_callback(
    callback: Callable[..., None],
    *args: Any,
    ordered: bool = False,
    partitioned: bool = False,
    **kwargs: Any,
) -> None: ...

class _DebugPrintFormatChecker(string.Formatter):
    def format_field(self, value, format_spec): ...
    def check_unused_args(self, used_args, args, kwargs) -> None: ...

formatter: Incomplete

def debug_print(
    fmt: str,
    *args,
    ordered: bool = False,
    partitioned: bool = False,
    skip_format_check: bool = False,
    _use_logging: bool = False,
    **kwargs,
) -> None: ...

debug_log: Incomplete
inspect_sharding_p: Incomplete
sharding_callbacks: Incomplete

class ShardingCallbackInfo:
    callback: Incomplete
    module_context: Incomplete
    def __init__(self, callback, module_context) -> None: ...

type Color = tuple[float, float, float] | str
type ColorMap = Callable[[float], tuple[float, float, float, float]]

def make_color_iter(color_map, num_rows, num_cols) -> Generator[Incomplete]: ...
def visualize_sharding(
    shape: Sequence[int],
    sharding: Sharding,
    *,
    use_color: bool = True,
    scale: float = 1.0,
    min_width: int = 9,
    max_width: int = 80,
    color_map: ColorMap | None = None,
): ...
def inspect_array_sharding(value, *, callback: Callable[[Sharding], None]): ...
def visualize_array_sharding(arr, **kwargs): ...
