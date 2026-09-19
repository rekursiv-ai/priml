from collections.abc import Callable, Sequence
from typing import Any

from _typeshed import Incomplete
from jax._src import (
    api as api,
    api_util as api_util,
    config as config,
    state as state,
    tree_util as tree_util,
    util as util,
)
from jax._src.frozen_dict import FrozenDict as FrozenDict
from jax._src.interpreters import mlir as mlir
from jax._src.pallas import (
    core as pallas_core,
    pallas_call as pallas_call,
)

mpmd_map_p: Incomplete

def mpmd_map(
    meshes_and_fns: Sequence[tuple[pallas_core.Mesh, Callable[_P, _T]]],
    /,
    out_shapes: tree_util.PyTree,
    *,
    scratch_shapes: pallas_core.ScratchShapeTree = (),
    compiler_params: Any | None = None,
    interpret: bool | Any = False,
    debug: bool = False,
    cost_estimate: pallas_core.CostEstimate | None = None,
    name: str | None = None,
    metadata: dict[str, str] | None = None,
) -> Callable[_P, _T]: ...
