from collections.abc import Sequence
from typing import Any

import dataclasses

from _typeshed import Incomplete
from jax._src import (
    core as jax_core,
    debugging as debugging,
    lax as lax,
    mesh as mesh_lib,
    source_info_util as source_info_util,
    state as state,
    tree_util as tree_util,
    util as util,
)
from jax._src.interpreters import mlir as mlir
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import (
    arith as arith,
    func as func,
    memref as memref,
    vector as vector,
)
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic import (
    core as tpu_core,
    lowering as tc_lowering,
    sc_core as sc_core,
)
from jax._src.state import indexing as indexing
from jax.experimental.mosaic.dialects import tpu as tpu

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
MemorySpace: Incomplete
LoweringContext: Incomplete
LoweringRuleContext: Incomplete

def dynamic_shape_replacement_fn(x): ...
def lower_jaxpr_to_module(
    lowering_context: mlir.LoweringRuleContext,
    grid_mapping: pallas_core.GridMapping,
    jaxpr: jax_core.Jaxpr,
    *,
    dimension_semantics: Sequence[tpu_core.DimensionSemantics] | None,
    kernel_type: tpu_core.CoreType,
    mesh: mesh_lib.Mesh | None = None,
    dynamic_shape_replacement_enabled: bool = False,
) -> ir.Module: ...
def lower_jaxpr_into_module(
    lowering_context: mlir.LoweringRuleContext,
    module: ir.Module,
    grid_mapping: pallas_core.GridMapping,
    jaxpr: jax_core.Jaxpr,
    *,
    name: str,
    dimension_semantics: Sequence[tpu_core.DimensionSemantics] | None,
    kernel_type: tpu_core.CoreType,
    mesh: mesh_lib.Mesh | None = None,
    dynamic_shape_replacement_enabled: bool = False,
) -> None: ...

@dataclasses.dataclass(init=False)
class MosaicGridMapping(tc_lowering.MosaicGridMapping):
    def __init__(
        self,
        jaxpr: jax_core.Jaxpr,
        grid_mapping: pallas_core.GridMapping,
        dimension_semantics: Sequence[tpu_core.DimensionSemantics] | None,
        mesh: mesh_lib.Mesh | None,
        kernel_type: tpu_core.CoreType,
    ) -> None: ...

def lower_jaxpr_to_func(
    jaxpr: jax_core.Jaxpr,
    *,
    name: str,
    kernel_type: tpu_core.CoreType,
    mosaic_grid_mapping: MosaicGridMapping,
    forward_compatible: bool,
    backend: Any | None,
) -> func.FuncOp: ...

register_lowering_rule: Incomplete
