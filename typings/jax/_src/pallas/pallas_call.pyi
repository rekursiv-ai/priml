from collections.abc import (
    Callable as Callable,
    Mapping,
)
from typing import Any

from _typeshed import Incomplete
from jax._src import (
    ad_util as ad_util,
    api as api,
    api_util as api_util,
    checkify as checkify,
    config as config,
    core as jax_core,
    effects as effects,
    hijax as hijax,
    state as state,
    tree_util as tree_util,
)
from jax._src.frozen_dict import FrozenDict as FrozenDict
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)
from jax._src.lib.mlir import ir as ir
from jax._src.mesh import get_abstract_mesh as get_abstract_mesh
from jax._src.pallas import (
    core as pallas_core,
    hlo_interpreter as hlo_interpreter,
    primitives as primitives,
)
from jax._src.shard_map import (
    P as P,
    shard_map as shard_map,
)
from jax._src.traceback_util import api_boundary as api_boundary
from jax._src.util import (
    safe_map as safe_map,
    safe_zip as safe_zip,
    split_list as split_list,
    tuple_insert as tuple_insert,
    unzip2 as unzip2,
)

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
BlockMapping: Incomplete
GridMapping: Incomplete
no_block_spec: Incomplete
CostEstimate: Incomplete
CompilerParams: Incomplete
pallas_call_p: Incomplete

def checkify_pallas_kernel_body_jaxpr(
    body_jaxpr: jax_core.ClosedJaxpr,
    enabled_errors,
    error: checkify.Error,
    grid_mapping: GridMapping,
) -> tuple[jax_core.ClosedJaxpr, tree_util.PyTreeDef, set[checkify.ErrorEffect]]: ...
def pallas_call_checkify_oob_grid(
    error: checkify.Error,
    enabled_errors,
    args: jax_core.Value,
    grid_mapping: GridMapping,
    input_output_aliases,
) -> checkify.Error: ...
def pallas_call_checkify_rule(
    error: checkify.Error,
    enabled_errors,
    *args: jax_core.Value,
    jaxpr: jax_core.Jaxpr,
    interpret: Any,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: GridMapping,
    out_avals: tuple[jax_core.AbstractValue, ...],
    **kwargs,
): ...
def pallas_call(
    kernel: Callable[..., None],
    out_shape: Any,
    *,
    grid_spec: pallas_core.GridSpec | None = None,
    grid: pallas_core.TupleGrid = (),
    in_specs: pallas_core.BlockSpecTree = ...,
    out_specs: pallas_core.BlockSpecTree = ...,
    scratch_shapes: pallas_core.ScratchShapeTree = (),
    input_output_aliases: Mapping[int, int] = {},
    debug: bool = False,
    interpret: Any = False,
    name: str | None = None,
    compiler_params: pallas_core.CompilerParams | None = None,
    cost_estimate: CostEstimate | None = None,
    metadata: dict[str, str] | None = None,
) -> Callable[..., Any]: ...
