from jax import dtypes as dtypes
from jax._src import (
    core as jax_core,
    frozen_dict as frozen_dict,
    sharding_impls as sharding_impls,
    tpu_custom_call as tpu_custom_call,
)
from jax._src.interpreters import mlir as mlir
from jax._src.lib.mlir import (
    ir as ir,
    passmanager as passmanager,
)
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic import (
    lowering as lowering,
    sc_lowering as sc_lowering,
)
from jax.experimental import mosaic as mosaic
from jax.experimental.mosaic.dialects import tpu as tpu

def pallas_call_tpu_lowering_rule(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    jaxpr: jax_core.Jaxpr,
    grid_mapping: pallas_core.GridMapping,
    mesh: pallas_core.Mesh | None,
    input_output_aliases: tuple[tuple[int, int], ...],
    debug: bool,
    interpret: bool,
    compiler_params: pallas_core.CompilerParams | None,
    cost_estimate: pallas_core.CostEstimate | None,
    out_avals: tuple[jax_core.AbstractValue, ...],
    metadata: frozen_dict.FrozenDict[str, str] | None,
    name: str | None,
): ...
def mpmd_map_tpu_lowering_rule(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    meshes,
    jaxprs,
    grid_mappings,
    out_avals,
    input_output_aliases,
    compiler_params,
    interpret,
    debug,
    cost_estimate,
    metadata,
    name,
): ...
