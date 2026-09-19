from jax._src import frozen_dict as frozen_dict
from jax._src.interpreters import mlir as mlir
from jax._src.lib import triton as triton
from jax._src.lib.mlir import ir as ir
from jax._src.pallas import core as pallas_core
from jax._src.pallas.triton import lowering as lowering

import jax._src.core as jax_core

def normalize_grid(grid: pallas_core.StaticGrid) -> tuple[int, int, int]: ...
def avals_to_layouts(avals): ...
def pallas_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
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
