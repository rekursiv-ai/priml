from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

from _typeshed import Incomplete
from jax._src import (
    core as jax_core,
    frozen_dict as frozen_dict,
    source_info_util as source_info_util,
    state as state,
    util as util,
)
from jax._src.lax import (
    lax as lax,
    slicing as slicing,
)
from jax._src.lax.control_flow import (
    conditionals as conditionals,
    loops as loops,
)
from jax._src.pallas import (
    core as pallas_core,
    primitives as primitives,
)
from jax._src.util import (
    foreach as foreach,
    safe_map as safe_map,
    safe_zip as safe_zip,
    split_list as split_list,
)

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
BlockMapping: Incomplete
GridMapping: Incomplete
CostEstimate: Incomplete

def kernel_to_hlo_jaxpr(
    jaxpr: jax_core.Jaxpr,
    consts: Sequence[Any],
    grid_mapping: GridMapping,
) -> tuple[jax_core.Jaxpr, Sequence[Any], Sequence[Any]]: ...
def eval_jaxpr_recursive(
    jaxpr: jax_core.Jaxpr,
    consts,
    *args,
    recurse_hop_rule: Callable[
        [jax_core.Jaxpr, Sequence[Any]],
        tuple[jax_core.Jaxpr, Sequence[Any]],
    ],
    propagate_source_info: bool = True,
) -> list[Any]: ...
def pad_jaxpr_constvars(
    jaxpr: jax_core.Jaxpr,
    i: int,
    all_const_avals: Sequence[Any],
) -> jax_core.ClosedJaxpr: ...
def make_hop_rule(primitive, *keys): ...
def resolve_physical_types(jaxpr: jax_core.Jaxpr, consts: Sequence[Any]): ...
def pallas_call_hlo_interpret(
    *args,
    jaxpr: jax_core.Jaxpr,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: GridMapping,
    mesh: pallas_core.Mesh | None,
    compiler_params: Any,
    cost_estimate: CostEstimate,
    out_avals: tuple[jax_core.AbstractValue, ...],
    metadata: frozen_dict.FrozenDict[str, str] | None,
    name: str | None,
): ...
