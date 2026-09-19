from collections.abc import (
    Callable as Callable,
    Generator,
    Sequence,
)
from typing import Any

import contextlib
import threading

from _typeshed import Incomplete
from jax import (
    export as export,
    tree_util as tree_util,
)
from jax._src import (
    api as api,
    api_util as api_util,
    config as config,
    core as core,
    dtypes as dtypes,
    op_shardings as op_shardings,
    source_info_util as source_info_util,
    util as util,
)
from jax._src.export import shape_poly as shape_poly
from jax._src.lib import xla_client as xla_client

NameStack: Incomplete
PolyShape: Incomplete
type DType = Any
DisabledSafetyCheck: Incomplete
map: Incomplete
zip: Incomplete
type TfVal = Any
PrecisionType = int

class _DefaultNativeSerialization: ...

DEFAULT_NATIVE_SERIALIZATION: Incomplete

class _ThreadLocalState(threading.local):
    inside_call_tf: bool
    shape_env: Sequence[tuple[str, TfVal]]
    call_tf_concrete_function_list: list[Any] | None
    def __init__(self) -> None: ...

@contextlib.contextmanager
def inside_call_tf() -> Generator[None]: ...
def get_thread_local_state_call_tf_concrete_function_list() -> list[Any] | None: ...
def convert(
    fun_jax: Callable,
    *,
    polymorphic_shapes: str
    | PolyShape
    | Sequence[str | PolyShape | None]
    | None = None,
    polymorphic_constraints: Sequence[str] = (),
    with_gradient: bool = True,
    enable_xla: bool = ...,
    native_serialization: bool | _DefaultNativeSerialization = ...,
    native_serialization_platforms: Sequence[str] | None = None,
    native_serialization_disabled_checks: Sequence[DisabledSafetyCheck] = (),
) -> Callable: ...

class NativeSerializationImpl:
    convert_kwargs: Incomplete
    fun_jax: Incomplete
    args_specs: Incomplete
    kwargs_specs: Incomplete
    native_serialization_disabled_checks: Incomplete
    native_serialization_platforms: Incomplete
    def __init__(
        self,
        fun_jax,
        *,
        args_specs,
        kwargs_specs,
        native_serialization_platforms: Sequence[str] | None,
        native_serialization_disabled_checks: Sequence[DisabledSafetyCheck],
    ) -> None: ...
    exported: Incomplete
    device_assignment: Incomplete
    def before_conversion(self) -> None: ...
    def after_conversion(self) -> None: ...
    def run_fun_tf(
        self,
        args_flat_tf: Sequence[TfVal],
    ) -> tuple[Sequence[TfVal], Sequence[core.ShapedArray], tree_util.PyTreeDef]: ...
    def get_vjp_fun(self) -> tuple[Callable, Sequence[core.AbstractValue]]: ...

def dtype_of_val(val: TfVal) -> DType: ...
def eval_polymorphic_shape(
    fun_jax: Callable,
    *,
    polymorphic_shapes=None,
) -> Callable: ...
def preprocess_arg_tf(arg_idx: int, arg_tf: TfVal) -> TfVal: ...

type PartitionsOrReplicated = tuple[int, ...] | None

def split_to_logical_devices(
    tensor: TfVal,
    partition_dimensions: PartitionsOrReplicated,
): ...
