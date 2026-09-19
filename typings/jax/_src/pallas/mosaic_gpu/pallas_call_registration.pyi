from jax._src import (
    core as jax_core,
    frozen_dict as frozen_dict,
    sharding_impls as sharding_impls,
)
from jax._src.interpreters import mlir as mlir
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic_gpu import lowering as lowering

def pallas_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *args,
    jaxpr: jax_core.Jaxpr,
    interpret: bool,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: pallas_core.GridMapping,
    mesh: pallas_core.Mesh | None,
    compiler_params: pallas_core.CompilerParams | None,
    cost_estimate: pallas_core.CostEstimate | None,
    out_avals: tuple[jax_core.AbstractValue, ...],
    metadata: frozen_dict.FrozenDict[str, str] | None,
    name: str | None,
): ...
