from collections.abc import Sequence
from typing import Any

import dataclasses

from _typeshed import Incomplete
from jax._src import (
    api_util as api_util,
    core as jax_core,
    custom_derivatives as custom_derivatives,
    pjit as pjit,
    tree_util as tree_util,
)
from jax._src.lax import lax as lax
from jax._src.pallas import core as pallas_core
from jax._src.state import discharge as discharge
from jax._src.util import (
    safe_map as safe_map,
    safe_zip as safe_zip,
)

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete

@dataclasses.dataclass(frozen=True)
class CostEstimate:
    flops: int
    transcendentals: int
    bytes_accessed: int
    def __add__(self, other: CostEstimate) -> CostEstimate: ...

def register_cost_rule(primitive: jax_core.Primitive, rule): ...

@dataclasses.dataclass(frozen=True)
class Context:
    avals_in: Sequence[Any]
    avals_out: Sequence[Any]

def cost_estimate_jaxpr(jaxpr: jax_core.ClosedJaxpr) -> pallas_core.CostEstimate: ...
def estimate_cost(fun, *args, **kwargs) -> pallas_core.CostEstimate: ...
def binary_cost_rule(ctx: Context, **_) -> CostEstimate: ...

BINARY_OPS: Incomplete

def unary_cost_rule(transcendental: bool): ...

UN_OPS: Incomplete
TRANSCENDENTAL_OPS: Incomplete

def dot_general_cost_rule(
    ctx: Context,
    dimension_numbers: lax.DotDimensionNumbers,
    **_,
) -> CostEstimate: ...
