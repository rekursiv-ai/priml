from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

import dataclasses

from _typeshed import Incomplete
from jax import (
    dlpack as dlpack,
    dtypes as dtypes,
    tree_util as tree_util,
)
from jax._src import (
    ad_util as ad_util,
    core as core,
    effects as effects,
    util as util,
)
from jax._src.interpreters import mlir as mlir
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import hlo as hlo
from jax.experimental import roofline as roofline

map: Incomplete
zip: Incomplete
type TfConcreteFunction = Any
TfVal: Incomplete

class UnspecifiedOutputShapeDtype: ...

def call_tf(
    callable_tf: Callable,
    has_side_effects: bool = True,
    ordered: bool = False,
    output_shape_dtype=...,
    call_tf_graph: bool = False,
) -> Callable: ...
def check_tf_result(
    idx: int,
    r_tf: TfVal,
    r_aval: core.ShapedArray | None,
) -> TfVal: ...

call_tf_p: Incomplete

@dataclasses.dataclass(frozen=True)
class CallTfEffect(effects.Effect): ...

call_tf_effect: Incomplete

class CallTfOrderedEffect(effects.Effect): ...

call_tf_ordered_effect: Incomplete

def emit_tf_embedded_graph_custom_call(
    ctx: mlir.LoweringRuleContext,
    concrete_function_flat_tf,
    operands: Sequence[ir.Value],
    has_side_effects,
    ordered,
    output_avals,
): ...
def add_to_call_tf_concrete_function_list(
    concrete_tf_fn: Any,
    call_tf_concrete_function_list: list[Any],
) -> int: ...
