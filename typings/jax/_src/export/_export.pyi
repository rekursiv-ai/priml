from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any, Protocol, TypeVar

import dataclasses

from _typeshed import Incomplete
from jax._src import (
    ad_util as ad_util,
    api as api,
    config as config,
    core as core,
    custom_derivatives as custom_derivatives,
    dispatch as dispatch,
    dtypes as dtypes,
    effects as effects,
    mesh as mesh_lib,
    pjit as pjit,
    sharding as sharding,
    sharding_impls as sharding_impls,
    source_info_util as source_info_util,
    stages as stages,
    traceback_util as traceback_util,
    tree_util as tree_util,
    typing as typing,
    util as util,
)
from jax._src.export import shape_poly as shape_poly
from jax._src.interpreters import (
    mlir as mlir,
    pxla as pxla,
)
from jax._src.lib import xla_client as xla_client
from jax._src.lib.mlir import (
    ir as ir,
    passmanager as passmanager,
)
from jax._src.lib.mlir.dialects import (
    hlo as hlo,
    sdy as sdy,
)

logger: Incomplete
map: Incomplete
zip: Incomplete
type DType = Any
Shape: Incomplete
LoweringSharding: Incomplete
NamedSharding: Incomplete
HloSharding: Incomplete
minimum_supported_calling_convention_version: int
maximum_supported_calling_convention_version: int

class DisabledSafetyCheck:
    @classmethod
    def platform(cls) -> DisabledSafetyCheck: ...
    @classmethod
    def custom_call(cls, target_name: str) -> DisabledSafetyCheck: ...
    def is_custom_call(self) -> str | None: ...
    def __init__(self, _impl: str) -> None: ...
    def __eq__(self, other) -> bool: ...
    def __hash__(self) -> int: ...

@dataclasses.dataclass(frozen=True)
class Exported:
    fun_name: str
    in_tree: tree_util.PyTreeDef
    in_avals: tuple[core.ShapedArray, ...]
    out_tree: tree_util.PyTreeDef
    out_avals: tuple[core.ShapedArray, ...]
    in_shardings_hlo: tuple[HloSharding | None, ...]
    out_shardings_hlo: tuple[HloSharding | None, ...]
    nr_devices: int
    platforms: tuple[str, ...]
    ordered_effects: tuple[effects.Effect, ...]
    unordered_effects: tuple[effects.Effect, ...]
    disabled_safety_checks: Sequence[DisabledSafetyCheck]
    mlir_module_serialized: bytes
    calling_convention_version: int
    module_kept_var_idx: tuple[int, ...]
    uses_global_constants: bool
    def mlir_module(self) -> str: ...
    def in_shardings_jax(
        self,
        mesh: mesh_lib.Mesh,
    ) -> Sequence[sharding.Sharding | None]: ...
    def out_shardings_jax(
        self,
        mesh: mesh_lib.Mesh,
    ) -> Sequence[sharding.Sharding | None]: ...
    def has_vjp(self) -> bool: ...
    def vjp(self) -> Exported: ...
    def serialize(self, vjp_order: int = 0) -> bytearray: ...
    def call(self, *args, **kwargs): ...

def deserialize(blob: bytearray) -> Exported: ...

T = TypeVar("T")
type PyTreeAuxData = Any

class _SerializeAuxData(Protocol):
    def __call__(self, aux_data: PyTreeAuxData, /) -> bytes: ...

class _DeserializeAuxData(Protocol):
    def __call__(self, serialized_aux_data: bytes, /) -> PyTreeAuxData: ...

class _BuildFromChildren(Protocol):
    def __call__(self, aux_data: PyTreeAuxData, children: Sequence[Any]) -> Any: ...

serialization_registry: dict[type, tuple[str, _SerializeAuxData]]
deserialization_registry: dict[
    str,
    tuple[type, _DeserializeAuxData, _BuildFromChildren],
]

def register_pytree_node_serialization(
    nodetype: type[T],
    *,
    serialized_name: str,
    serialize_auxdata: _SerializeAuxData,
    deserialize_auxdata: _DeserializeAuxData,
    from_children: _BuildFromChildren | None = None,
) -> type[T]: ...
def register_namedtuple_serialization(
    nodetype: type[T],
    *,
    serialized_name: str,
) -> type[T]: ...
def default_export_platform() -> str: ...

default_lowering_platform = default_export_platform

def shape_and_dtype_jax_array(a) -> tuple[Sequence[int | None], DType]: ...
def export(
    fun_jit: stages.Wrapped,
    *,
    platforms: Sequence[str] | None = None,
    disabled_checks: Sequence[DisabledSafetyCheck] = (),
    _override_lowering_rules: Sequence[tuple[Any, Any]] | None = None,
) -> Callable[..., Exported]: ...
def check_symbolic_scope_errors(fun_jax, args_specs, kwargs_specs) -> None: ...
def to_named_sharding_with_abstract_mesh(
    s: LoweringSharding,
    aval: core.ShapedArray,
    mesh: mesh_lib.Mesh | mesh_lib.AbstractMesh | None,
) -> NamedSharding | None: ...
def named_to_hlo_sharding(
    s: NamedSharding | None,
    aval: core.ShapedArray,
) -> HloSharding | None: ...

check_sharding_pattern: Incomplete

def expand_in_shardings(
    in_shardings: Sequence[LoweringSharding],
    module_kept_var_idx: Sequence[int],
    nr_inputs: int,
) -> Sequence[LoweringSharding]: ...
def call(exported: Exported) -> Callable[..., typing.Array]: ...

call_exported = call
call_exported_p: Incomplete

def get_mesh_from_symbol(symtab: ir.SymbolTable) -> mesh_lib.AbstractMesh: ...
def has_sdy_meshes_in_frontend_attributes(submodule: ir.Module) -> bool: ...
def has_sdy_mesh(symtab: ir.SymbolTable, submodule: ir.Module) -> bool: ...
def wrap_with_sharding(
    ctx: mlir.LoweringRuleContext,
    x: ir.Value,
    x_aval: core.AbstractValue,
    x_sharding: sharding_impls.NamedSharding
    | sharding_impls.GSPMDSharding
    | HloSharding
    | None,
    use_shardy: bool,
) -> ir.Value: ...
