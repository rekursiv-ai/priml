from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

import dataclasses

from _typeshed import Incomplete
from jax._src import (
    api as api,
    config as config,
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
    effects as effects,
    ffi as ffi,
    pickle_util as pickle_util,
    sharding_impls as sharding_impls,
    tree_util as tree_util,
    util as util,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import hlo as hlo
from jax._src.sharding_impls import (
    SdyArray as SdyArray,
    SdyArrayList as SdyArrayList,
    SdyDim as SdyDim,
    SingleDeviceSharding as SingleDeviceSharding,
)
from jax._src.typing import (
    Array as Array,
    DeprecatedArg as DeprecatedArg,
)

logger: Incomplete
pure_callback_p: Incomplete
map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete

@dataclasses.dataclass(frozen=True)
class _FlatCallback:
    callback_func: Callable[..., Any]
    in_tree: tree_util.PyTreeDef
    def __call__(self, *flat_args: Array) -> Sequence[Array]: ...

def pure_callback_impl(
    *args,
    result_avals,
    callback: _FlatCallback,
    sharding: SingleDeviceSharding | None,
    vmap_method: str | None,
): ...
@pure_callback_p.def_abstract_eval
def pure_callback_abstract_eval(
    *avals,
    callback: _FlatCallback,
    result_avals,
    sharding: SingleDeviceSharding | None,
    vmap_method: str | None,
): ...
def pure_callback_jvp_rule(*args, **kwargs) -> None: ...
def pure_callback_transpose_rule(*args, **kwargs) -> None: ...
def pure_callback_lowering(
    ctx,
    *args,
    callback: _FlatCallback,
    sharding: SingleDeviceSharding | None,
    **params,
): ...
def pure_callback(
    callback: Callable[..., Any],
    result_shape_dtypes: Any,
    *args: Any,
    sharding: SingleDeviceSharding | None = None,
    vectorized: bool | DeprecatedArg | None = ...,
    vmap_method: str | None = None,
    **kwargs: Any,
): ...

io_callback_p: Incomplete

class IOEffect(effects.Effect): ...
class OrderedIOEffect(effects.Effect): ...

def io_callback_impl(
    *args,
    result_avals,
    callback: _FlatCallback,
    sharding: SingleDeviceSharding | None,
    ordered: bool,
): ...
@io_callback_p.def_effectful_abstract_eval
def io_callback_abstract_eval(
    *avals,
    callback: _FlatCallback,
    result_avals,
    sharding: SingleDeviceSharding | None,
    ordered: bool,
): ...
def io_callback_jvp_rule(*args, **kwargs) -> None: ...
def io_callback_transpose_rule(*args, **kwargs) -> None: ...
def io_callback_batching_rule(
    args,
    dims,
    callback,
    result_avals,
    sharding,
    ordered,
): ...
def io_callback_lowering(ctx, *args, callback, sharding, ordered, **params): ...
def io_callback(
    callback: Callable[..., Any],
    result_shape_dtypes: Any,
    *args: Any,
    sharding: SingleDeviceSharding | None = None,
    ordered: bool = False,
    **kwargs: Any,
): ...
def is_empty_shape(s: core.Shape) -> bool: ...
def send_to_host(
    channel: int,
    token: hlo.TokenType,
    operand: Any,
    name: str | None = None,
    *,
    sharding: SdyArrayList | xc.OpSharding | None = None,
) -> ir.Value: ...
def receive_from_host(
    channel: int,
    token: hlo.TokenType,
    out_aval: core.ShapedArray,
    name: str | None = None,
    *,
    sharding: SdyArrayList | xc.OpSharding | None = None,
) -> tuple[ir.Value, ir.Value]: ...
def emit_python_callback(
    ctx: mlir.LoweringRuleContext,
    callback,
    token: Any | None,
    operands: Sequence[ir.Value],
    operand_avals: Sequence[core.ShapedArray],
    result_avals: Sequence[core.ShapedArray],
    *,
    has_side_effect: bool,
    returns_token: bool = True,
    partitioned: bool = False,
    sharding: SdyArrayList | xc.OpSharding | None = None,
) -> tuple[Sequence[mlir.IrValues], Any, Any]: ...
