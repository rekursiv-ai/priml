from collections.abc import Sequence
from typing import Any

import atexit
import dataclasses
import threading
import types

from _typeshed import Incomplete
from jax._src import (
    api as api,
    array as array,
    basearray as basearray,
    config as config,
    core as core,
    dtypes as dtypes,
    literals as literals,
    pjit as pjit,
    traceback_util as traceback_util,
    util as util,
    xla_bridge as xla_bridge,
)
from jax._src.abstract_arrays import array_types as array_types
from jax._src.api_util import InternalFloatingPointError as InternalFloatingPointError
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
    partial_eval as partial_eval,
    pxla as pxla,
)
from jax._src.layout import (
    Format as Format,
    Layout as Layout,
)
from jax._src.mesh import (
    AbstractMesh as AbstractMesh,
    Mesh as Mesh,
)
from jax._src.monitoring import (
    record_event_duration_secs as record_event_duration_secs,
    record_event_time_span as record_event_time_span,
    record_scalar as record_scalar,
)
from jax._src.partition_spec import PartitionSpec as PartitionSpec
from jax._src.sharding import Sharding as Sharding
from jax._src.sharding_impls import (
    GSPMDSharding as GSPMDSharding,
    NamedSharding as NamedSharding,
    SingleDeviceSharding as SingleDeviceSharding,
    is_single_device_sharding as is_single_device_sharding,
)
from jax._src.stages import SourceInfo as SourceInfo

JAXPR_TRACE_EVENT: str
JAXPR_TO_MLIR_MODULE_EVENT: str
BACKEND_COMPILE_EVENT: str
xe: Incomplete
Backend: Incomplete
Device: Incomplete
ArrayCopySemantics: Incomplete
CompileOptions: Incomplete
map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
logger: Incomplete

def apply_primitive(prim, *args, **params): ...
def xla_primitive_callable(prim: core.Primitive, **params): ...
def simple_impl(prim) -> None: ...

type RuntimeToken = Any

class RuntimeTokenSet(threading.local):
    current_tokens: dict[core.Effect, core.Token]
    output_runtime_tokens: dict[Device, RuntimeToken]
    def __init__(self) -> None: ...
    def get_token_input(
        self,
        eff: core.Effect,
        devices: list[Device],
    ) -> core.Token: ...
    def set_token_result(self, eff: core.Effect, token: core.Token): ...
    def set_output_runtime_token(self, device: Device, token: RuntimeToken): ...
    def clear(self) -> None: ...
    def block_until_ready(self) -> None: ...

runtime_tokens: RuntimeTokenSet

@atexit.register
def wait_for_tokens() -> None: ...

class LogElapsedTimeContextManager:
    fmt: Incomplete
    fun_name: Incomplete
    event: Incomplete
    def __init__(self, fmt: str, fun_name: str, event: str | None = None) -> None: ...
    start_time: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...

log_elapsed_time = LogElapsedTimeContextManager

def should_tuple_args(num_args: int, platform: str) -> bool: ...
def jaxpr_has_primitive(jaxpr: core.Jaxpr, prim_name: str) -> bool: ...

prim_requires_devices_during_lowering: set[core.Primitive]

@util.weakref_lru_cache
def jaxpr_has_prim_requiring_devices(jaxpr: core.Jaxpr) -> bool: ...
@util.weakref_lru_cache
def get_intermediate_shardings(
    jaxpr: core.Jaxpr,
) -> Sequence[tuple[Sharding, SourceInfo]]: ...
def check_arg(arg: Any): ...
def needs_check_special() -> bool: ...
def check_special(name: str, bufs: Sequence[basearray.Array]) -> None: ...
def check_special_array(name: str, arr: array.ArrayImpl) -> array.ArrayImpl: ...

@dataclasses.dataclass(frozen=True)
class _DeferredShardArg:
    x: Any
    s: Sharding
    aval: core.AbstractValue
    committed: bool
    copy_semantics: ArrayCopySemantics
    def result_handler(self, shard_arg_result): ...

@dataclasses.dataclass(frozen=True)
class _DeferredCrossHostTransferArg:
    x: array.ArrayImpl
    dst_sharding: Sharding
    copy_semantics: ArrayCopySemantics

def batched_device_put_impl(
    *xs,
    devices: Sequence[Device | Sharding | Format | None],
    srcs: Sequence[Device | Sharding | Format | None],
    copy_semantics: Sequence[ArrayCopySemantics],
): ...

device_put_p: Incomplete

def update_dp_aval(aval, d): ...
