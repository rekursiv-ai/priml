from collections.abc import Generator
from typing import Any

import contextlib

from _typeshed import Incomplete
from jax._src import (
    array as array,
    core as core,
    distributed as distributed,
    dtypes as dtypes,
    prng as prng,
    sharding_impls as sharding_impls,
    xla_bridge as xla_bridge,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
    pxla as pxla,
)
from jax._src.lib import xla_client as xla_client
from jax._src.util import safe_zip as safe_zip
from jax.tree_util import (
    tree_flatten as tree_flatten,
    tree_unflatten as tree_unflatten,
)

import jax

def broadcast_one_to_all(in_tree: Any, is_source: bool | None = None) -> Any: ...
def process_allgather(in_tree: Any, tiled: bool = False) -> Any: ...
def sync_global_devices(name: str): ...
def assert_equal(in_tree, fail_message: str = ""): ...
def reached_preemption_sync_point(step_id: int) -> bool: ...
def host_local_array_to_global_array_impl(
    arr: Any,
    *,
    global_mesh: jax.sharding.Mesh,
    pspec: Any,
): ...
def host_local_array_to_global_array(
    local_inputs: Any,
    global_mesh: jax.sharding.Mesh,
    pspecs: Any,
): ...

host_local_array_to_global_array_p: Incomplete

def ltg_abstract_eval(arr, *, global_mesh, pspec): ...
def ltg_batcher(insert_axis, axis_data, vals_in, dims_in, global_mesh, pspec): ...
def global_array_to_host_local_array_impl(
    arr: Any,
    *,
    global_mesh: jax.sharding.Mesh,
    pspec: Any,
): ...
def global_array_to_host_local_array(
    global_inputs: Any,
    global_mesh: jax.sharding.Mesh,
    pspecs: Any,
): ...

global_array_to_host_local_array_p: Incomplete

def gtl_abstract_eval(arr, *, global_mesh, pspec): ...

class _LiveDevices:
    devices: Incomplete
    def __init__(self) -> None: ...
    @contextlib.contextmanager
    def __call__(self, devices) -> Generator[Incomplete, None, Incomplete]: ...

live_devices: Incomplete
