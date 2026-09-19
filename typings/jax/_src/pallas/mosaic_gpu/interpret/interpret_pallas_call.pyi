from collections.abc import (
    Mapping,
    Sequence,
    Set as AbstractSet,
)
from typing import Any

import dataclasses

from _typeshed import Incomplete
from jax._src import (
    callback as callback,
    core as jax_core,
    effects as effects,
)
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic.interpret import (
    thread_map as thread_map,
    utils as interpret_utils,
)
from jax._src.pallas.mosaic_gpu.interpret import (
    gpu_callbacks as gpu_callbacks,
    jaxpr_interpret as jaxpr_interpret,
)
from jax._src.typing import Array as Array
from jax._src.util import (
    safe_zip as safe_zip,
    split_list as split_list,
)
from jax.experimental.pallas import mosaic_gpu as plgpu

import jax

InterpretParams: Incomplete

def get_interpret_effects() -> AbstractSet[effects.Effect]: ...
def get_races() -> gpu_callbacks.RaceDetectionState: ...
def reset_gpu_interpret_mode_state() -> None: ...

@dataclasses.dataclass(frozen=True)
class AllocationKeyAndValue:
    key: jax.Array
    value: jax.Array
    @property
    def shape(self) -> tuple[int, ...]: ...

def interpret_pallas_call(
    *args,
    jaxpr: jax_core.Jaxpr,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: pallas_core.GridMapping,
    mesh: plgpu.Mesh | None,
    compiler_params: Mapping[str, Any],
    cost_estimate: pallas_core.CostEstimate,
    out_avals: tuple[jax_core.AbstractValue, ...],
    interpret_params: interpret_utils.InterpretGPUParams,
    metadata: Mapping[str, str] | None,
    **kwargs,
) -> Sequence[Array]: ...
