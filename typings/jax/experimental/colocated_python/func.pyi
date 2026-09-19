from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

import dataclasses
import inspect

from _typeshed import Incomplete
from jax._src import (
    api as api,
    tree_util as tree_util,
    util as util,
)
from jax._src.interpreters import pxla as pxla
from jax._src.lib import xla_client as xc
from jax._src.traceback_util import api_boundary as api_boundary
from jax._src.util import wraps as wraps
from jax.experimental.colocated_python import func_backend as func_backend
from jax.extend.ifrt_programs import ifrt_programs as ifrt_programs

import jax

type ShapeDtypeStructTree = Any

@dataclasses.dataclass(frozen=True, slots=True)
class FunctionInfo:
    fun: Callable[..., Any]
    fun_sourceinfo: str | None
    fun_signature: inspect.Signature | None

@dataclasses.dataclass(frozen=True, slots=True)
class Specialization:
    in_specs_treedef: tree_util.PyTreeDef | None = ...
    in_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = ...
    out_specs_fn: Callable[..., ShapeDtypeStructTree] | None = ...
    out_specs_treedef: tree_util.PyTreeDef | None = ...
    out_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = ...
    devices: xc.DeviceList | None = ...
    def update(
        self,
        *,
        in_specs_treedef: tree_util.PyTreeDef | None = None,
        in_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = None,
        out_specs_fn: Callable[..., ShapeDtypeStructTree] | None = None,
        out_specs_treedef: tree_util.PyTreeDef | None = None,
        out_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = None,
        devices: Sequence[jax.Device] | xc.DeviceList | None = None,
    ): ...

class _SpecializedCollection:
    @dataclasses.dataclass(slots=True, unsafe_hash=True)
    class WeakSpec:
        dtypes: tuple[jax.numpy.dtype, ...]
        shapes: tuple[tuple[int, ...], ...]
        sharding_ids: tuple[int, ...]
        treedef: tree_util.PyTreeDef
        def __init__(
            self,
            args_leaves: Sequence[jax.Array],
            treedef: tree_util.PyTreeDef,
        ) -> None: ...

    @dataclasses.dataclass(slots=True, unsafe_hash=True)
    class StrongSpec:
        in_specs_treedef: tree_util.PyTreeDef | None = ...
        in_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = ...
        def __init__(
            self,
            args_leaves: Sequence[jax.Array],
            pytreedef: tree_util.PyTreeDef,
        ) -> None: ...

    def __init__(self) -> None: ...
    def get(
        self,
        args_leaves: Sequence[jax.Array],
        pytreedef: tree_util.PyTreeDef,
        func_info: FunctionInfo,
        specialization: Specialization,
    ) -> Callable[..., Any]: ...

class _JaxSecondLevelCaches:
    def __init__(self, name: str) -> None: ...
    def cache_clear(self) -> None: ...
    def register_second_level(
        self,
        uid: int,
        cache_clear_callback: Callable[..., Any],
    ): ...
    def remove_second_level(self, uid: int): ...

class _CachedColocatedFunctionMaker:
    JAX_CACHE: Incomplete
    held_by: Incomplete
    specialized_collections: Incomplete
    specialized_functions: Incomplete
    def __init__(self, held_by: int | None) -> None: ...
    def __del__(self) -> None: ...
    def make_callable(
        self,
        fun: Callable[..., Any],
        fun_sourceinfo: str | None,
        fun_signature: inspect.Signature | None,
    ): ...

def make_callable(
    fun: Callable[..., Any],
    fun_sourceinfo: str | None,
    fun_signature: inspect.Signature | None,
): ...
