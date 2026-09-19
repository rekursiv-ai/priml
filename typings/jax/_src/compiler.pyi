from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

from _typeshed import Incomplete
from jax._src import (
    compilation_cache as compilation_cache,
    distributed as distributed,
    lib as lib,
    monitoring as monitoring,
    profiler as profiler,
    traceback_util as traceback_util,
    util as util,
)
from jax._src.interpreters import mlir as mlir
from jax._src.lib import (
    _jax,
    xla_client as xc,
)
from jax._src.lib.mlir import ir as ir

import numpy as np

CompileOptions: Incomplete
logger: Incomplete

def get_latest_profile_version(backend: xc.Client) -> int: ...
def use_detailed_logging(module: ir.Module) -> bool: ...
def log_persistent_cache_hit(module_name: str, cache_key: str) -> None: ...
def log_persistent_cache_miss(module_name: str, cache_key: str) -> None: ...
def get_compile_options(
    num_replicas: int,
    num_partitions: int,
    device_assignment=None,
    use_spmd_partitioning: bool = True,
    use_auto_spmd_partitioning: bool = False,
    auto_spmd_partitioning_mesh_shape: list[int] | None = None,
    auto_spmd_partitioning_mesh_ids: list[int] | None = None,
    env_options_overrides: dict[str, str] | None = None,
    fdo_profile: bytes | None = None,
    detailed_logging: bool = True,
    backend: xc.Client | None = None,
) -> xc.CompileOptions: ...
@profiler.annotate_function
def backend_compile(
    backend: xc.Client,
    module: ir.Module,
    executable_devices: xc.DeviceList,
    options: xc.CompileOptions,
) -> xc.Executable: ...
@profiler.annotate_function
def backend_compile_and_load(
    backend: xc.Client,
    module: ir.Module,
    executable_devices: xc.DeviceList,
    options: xc.CompileOptions,
    host_callbacks: Sequence[Any],
) -> xc.LoadedExecutable: ...
def register_xla_runtime_error_handler(
    handler_fn: Callable[[_jax.JaxRuntimeError], Exception | None],
): ...
def compile_or_get_cached(
    backend: xc.Client,
    computation: ir.Module,
    devices: np.ndarray,
    compile_options: xc.CompileOptions,
    host_callbacks: Sequence[Any],
    executable_devices: xc.DeviceList,
    pgle_profiler: profiler.PGLEProfiler | None = None,
) -> xc.LoadedExecutable: ...
