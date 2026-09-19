from collections.abc import (
    Callable as Callable,
    Mapping,
)
from typing import Any

import dataclasses
import functools

from jax import lax as lax
from jax._src import (
    core as jax_core,
    source_info_util as source_info_util,
)
from jax._src.pallas import primitives as primitives
from jax._src.pallas.mosaic.interpret import utils as interpret_utils
from jax._src.pallas.mosaic_gpu.interpret import gpu_callbacks as gpu_callbacks
from jax._src.state import indexing as indexing
from jax._src.util import (
    safe_zip as safe_zip,
    split_list as split_list,
)
from jax.experimental.pallas import mosaic_gpu as plgpu

@dataclasses.dataclass(init=False, frozen=True)
class DeviceInfo:
    axis_indices: Mapping[jax_core.AxisName, int]
    axis_sizes: Mapping[jax_core.AxisName, int]
    def __init__(self) -> None: ...
    @functools.cached_property
    def device_id(self) -> int: ...
    @functools.cached_property
    def num_devices(self) -> int: ...

@dataclasses.dataclass(frozen=True, kw_only=True)
class JaxprInterpreter:
    grid_point_coords: tuple[int]
    thread_id: int
    mesh: plgpu.Mesh | None
    device_info: DeviceInfo
    compiler_params: Mapping[str, Any]
    interpret_params: interpret_utils.InterpretParams
    @functools.cached_property
    def num_threads(self) -> int: ...
    def interpret(self, jaxpr, *args): ...
