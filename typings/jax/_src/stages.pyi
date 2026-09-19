from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import dataclasses
import enum

from _typeshed import Incomplete
from jax._src import (
    config as config,
    core as core,
    sharding as sharding_lib,
    source_info_util as source_info_util,
    traceback_util as traceback_util,
    tree_util as tree_util,
    util as util,
)
from jax._src.core import typeof as typeof
from jax._src.interpreters import mlir as mlir
from jax._src.layout import (
    AutoLayout as AutoLayout,
    Format as Format,
    Layout as Layout,
)
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir as ir
from jax._src.sharding_impls import (
    AUTO as AUTO,
    UnspecifiedValue as UnspecifiedValue,
)
from jax._src.tree_util import (
    tree_structure as tree_structure,
    tree_unflatten as tree_unflatten,
)
from jax._src.typing import ArrayLike as ArrayLike

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
type CompilerOptions = dict[str, str | bool]

class Executable:
    def xla_extension_executable(self) -> xc.LoadedExecutable: ...
    def call(self, *args_flat) -> Sequence[Any]: ...
    def create_cpp_call(self, params: CompiledCallParams) -> Any: ...
    def input_shardings(self) -> Sequence[sharding_lib.Sharding]: ...
    def output_shardings(self) -> Sequence[sharding_lib.Sharding]: ...
    def input_formats(self) -> None: ...
    def output_formats(self) -> None: ...
    def as_text(self) -> str: ...
    def cost_analysis(self) -> Any: ...
    def memory_analysis(self) -> Any: ...
    def runtime_executable(self) -> Any: ...

class Lowering:
    compile_args: dict[str, Any]
    const_args: list[ArrayLike]
    def hlo(self) -> xc.XlaComputation: ...
    def stablehlo(self) -> ir.Module: ...
    def compile(
        self,
        compiler_options: CompilerOptions | None = None,
        *,
        device_assignment: tuple[xc.Device, ...] | None = None,
    ) -> Executable: ...
    def as_text(
        self,
        dialect: str | None = None,
        *,
        debug_info: bool = False,
    ) -> str: ...
    def compiler_ir(self, dialect: str | None = None) -> Any: ...
    def cost_analysis(self) -> Any: ...

@dataclass(frozen=True)
class ArgInfo:
    donated: bool
    @property
    def shape(self): ...
    @property
    def dtype(self): ...

class Stage:
    args_info: Any
    @property
    def in_tree(self) -> tree_util.PyTreeDef: ...
    @property
    def in_avals(self): ...
    @property
    def donate_argnums(self): ...

def make_args_info(in_tree, in_avals, donate_argnums): ...

class CompiledCallParams(NamedTuple):
    executable: Executable
    no_kwargs: bool
    in_tree: tree_util.PyTreeDef
    out_tree: tree_util.PyTreeDef
    const_args: list[ArrayLike]
    in_types: tuple[tree_util.PyTreeDef, list[core.AbstractValue | core.AvalQDD]] | None
    out_types: tuple[tree_util.PyTreeDef, list[core.AbstractValue]] | None
    @property
    def is_high(self): ...

class Traced(Stage):
    out_tree: Incomplete
    def __init__(self, meta_tys_flat, params, in_tree, out_tree, consts) -> None: ...
    jaxpr: Incomplete
    fun_name: Incomplete
    args_info: Incomplete
    out_info: Incomplete
    @property
    def out_avals(self): ...
    def __call__(self, *args, **kwargs): ...
    @property
    def lojax(self) -> LoJax: ...
    def lower(
        self,
        *,
        lowering_platforms: tuple[str, ...] | None = None,
        _private_parameters: mlir.LoweringParameters | None = None,
    ): ...

def lojax_expand_params(jaxpr, params): ...
def lojax_pytree(hi_avals, tree): ...

class LoJax:
    out_tree: Incomplete
    def __init__(
        self,
        meta_tys_flat,
        params,
        in_tree,
        out_tree,
        in_types,
        out_types,
        consts,
    ) -> None: ...
    jaxpr: Incomplete
    fun_name: Incomplete
    args_info: Incomplete
    out_info: Incomplete

class Lowered(Stage):
    args_info: Any
    out_tree: tree_util.PyTreeDef
    def __init__(
        self,
        lowering: Lowering,
        args_info,
        out_tree: tree_util.PyTreeDef,
        no_kwargs: bool = False,
        in_types=None,
        out_types=None,
    ) -> None: ...
    @property
    def in_avals(self): ...
    @property
    def out_info(self): ...
    def compile(
        self,
        compiler_options: CompilerOptions | None = None,
        *,
        device_assignment: tuple[xc.Device, ...] | None = None,
    ) -> Compiled: ...
    def as_text(
        self,
        dialect: str | None = None,
        *,
        debug_info: bool = False,
    ) -> str: ...
    def compiler_ir(self, dialect: str | None = None) -> None: ...
    def cost_analysis(self) -> None: ...

class Compiled(Stage):
    args_info: Any
    out_tree: tree_util.PyTreeDef
    def __init__(
        self,
        executable,
        const_args: list[ArrayLike],
        args_info,
        out_tree,
        no_kwargs: bool = False,
        in_types=None,
        out_types=None,
    ) -> None: ...
    def as_text(self) -> str | None: ...
    def cost_analysis(self) -> None: ...
    def memory_analysis(self) -> None: ...
    @property
    def in_avals(self): ...
    @property
    def out_info(self): ...
    def runtime_executable(self) -> None: ...
    @property
    def input_shardings(self): ...
    @property
    def output_shardings(self): ...
    @property
    def input_formats(self): ...
    @property
    def output_formats(self): ...
    @staticmethod
    def call(*args, **kwargs): ...
    def __call__(self, *args, **kwargs): ...

class Wrapped(Protocol):
    def __call__(self, *args, **kwargs): ...
    def trace(self, *args, **kwargs) -> Traced: ...
    def lower(self, *args, **kwargs) -> Lowered: ...

class MismatchType(enum.Enum):
    ARG_SHARDING = ...
    CONST_SHARDING = ...
    OUT_SHARDING = ...
    SHARDING_INSIDE_COMPUTATION = ...
    CONTEXT_DEVICES = ...
    IN_SHARDING = ...

class SourceInfo(NamedTuple):
    source_info: source_info_util.SourceInfo
    eqn_name: str

@dataclasses.dataclass
class DeviceAssignmentMismatch:
    da: Sequence[xc.Device]
    m_type: MismatchType
    source_info: SourceInfo | None
    @property
    def device_ids(self) -> Sequence[int]: ...
    @property
    def platform(self) -> str: ...
    @property
    def source_info_str(self): ...
    def m_type_str(self, api_name): ...

class DeviceAssignmentMismatchError(Exception): ...
