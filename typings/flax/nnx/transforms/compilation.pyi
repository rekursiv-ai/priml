import dataclasses
import typing as tp

from _typeshed import Incomplete
from flax.nnx import (
    extract as extract,
    filterlib as filterlib,
    graphlib as graphlib,
    statelib as statelib,
    variablelib as variablelib,
)
from flax.typing import (
    MISSING as MISSING,
    Missing as Missing,
    PathParts as PathParts,
)
from jax.sharding import (
    AbstractMesh as AbstractMesh,
    Mesh as Mesh,
)

import jax

F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
P = tp.ParamSpec("P")
R = tp.TypeVar("R")
type Specs = tp.Any
AxisName = tp.Hashable

class StateSharding(extract.PrefixMapping):
    def __init__(
        self,
        filter_sharding: statelib.State
        | tp.Mapping[filterlib.Filter, tp.Any]
        | tp.Iterable[tuple[filterlib.Filter, tp.Any]],
        /,
    ) -> None: ...
    @property
    def filters(self) -> tuple[filterlib.Filter, ...]: ...
    @property
    def shardings(self) -> tuple[tp.Any, ...]: ...
    def map_prefix(self, path: PathParts, variable: variablelib.Variable) -> tp.Any: ...
    def __eq__(self, other): ...
    def __hash__(self): ...

@dataclasses.dataclass(eq=False)
class JitFn:
    f: tp.Callable[..., tp.Any]
    in_shardings: tp.Any
    out_shardings: tp.Any
    kwarg_shardings: tp.Any
    ctxtag: tp.Hashable
    def __post_init__(self) -> None: ...
    def __call__(self, *pure_args, **pure_kwargs): ...

@tp.overload
def jit(
    *,
    in_shardings: tp.Any = None,
    out_shardings: tp.Any = None,
    static_argnums: int | tp.Sequence[int] | None = None,
    static_argnames: str | tp.Iterable[str] | None = None,
    donate_argnums: int | tp.Sequence[int] | None = None,
    donate_argnames: str | tp.Iterable[str] | None = None,
    keep_unused: bool = False,
    device: jax.Device | None = None,
    backend: str | None = None,
    inline: bool = False,
    graph: bool | None = None,
) -> tp.Callable[[tp.Callable[P, R]], JitWrapped[P, R]]: ...
@tp.overload
def jit(
    fun: tp.Callable[P, R],
    *,
    in_shardings: tp.Any = None,
    out_shardings: tp.Any = None,
    static_argnums: int | tp.Sequence[int] | None = None,
    static_argnames: str | tp.Iterable[str] | None = None,
    donate_argnums: int | tp.Sequence[int] | None = None,
    donate_argnames: str | tp.Iterable[str] | None = None,
    keep_unused: bool = False,
    device: jax.Device | None = None,
    backend: str | None = None,
    inline: bool = False,
    graph: bool | None = None,
) -> JitWrapped[P, R]: ...

@dataclasses.dataclass(frozen=True, slots=True)
class PartialState:
    treedef: jax.tree_util.PyTreeDef
    leaves: list[tp.Any]

@dataclasses.dataclass(eq=False)
class TreeJitFn:
    f: tp.Callable[..., tp.Any]
    donate_argnums: frozenset[int]
    donate_argnames: frozenset[str]
    def __post_init__(self) -> None: ...
    @extract.treemap_copy_args
    def __call__(self, *args, **kwargs): ...

class TreeJitWrapped(tp.Generic[P, R]):
    fun: tp.Callable[P, R]
    in_shardings: Incomplete
    out_shardings: Incomplete
    partial_args: Incomplete
    jitted_fn: Incomplete
    def __init__(
        self,
        fun: tp.Callable[P, R],
        in_shardings: tp.Any,
        out_shardings: tp.Any,
        static_argnums: int | tp.Sequence[int] | None = None,
        static_argnames: str | tp.Iterable[str] | None = None,
        donate_argnums: int | tp.Sequence[int] | None = None,
        donate_argnames: str | tp.Iterable[str] | None = None,
        keep_unused: bool = False,
        device: jax.Device | None = None,
        backend: str | None = None,
        inline: bool = False,
        partial_args: tuple[PartialState, ...] = (),
    ) -> None: ...
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    def __get__(self, obj, objtype=None): ...
    def eval_shape(self, *args, **kwargs): ...
    def trace(self, *args, **kwargs) -> TreeTraced: ...
    def lower(self, *args, **kwargs) -> TreeLowered: ...

def jit_partial(
    fun: tp.Callable[..., R],
    *partial_args: tp.Any,
    in_shardings: tp.Any = None,
    out_shardings: tp.Any = None,
    donate_argnums: int | tp.Sequence[int] | None = None,
    donate_argnames: str | tp.Iterable[str] | None = None,
    keep_unused: bool = False,
    device: jax.Device | None = None,
    backend: str | None = None,
    inline: bool = False,
    graph: bool | None = None,
) -> TreeJitWrapped[..., R]: ...

class JitWrapped(tp.Generic[P, R]):
    fun: tp.Callable[P, R]
    jax_in_shardings: Incomplete
    jax_out_shardings: Incomplete
    jitted_fn: Incomplete
    in_shardings: Incomplete
    out_shardings: Incomplete
    kwarg_shardings: Incomplete
    static_argnums: Incomplete
    def __init__(
        self,
        fun: tp.Callable[P, R],
        in_shardings: tp.Any,
        out_shardings: tp.Any,
        static_argnums: int | tp.Sequence[int] | None = None,
        static_argnames: str | tp.Iterable[str] | None = None,
        donate_argnums: int | tp.Sequence[int] | None = None,
        donate_argnames: str | tp.Iterable[str] | None = None,
        keep_unused: bool = False,
        device: jax.Device | None = None,
        backend: str | None = None,
        inline: bool = False,
    ) -> None: ...
    def __get__(self, obj, objtype=None): ...
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    def eval_shape(self, *args, **kwargs): ...
    def trace(self, *args, **kwargs) -> Traced: ...
    def lower(self, *args, **kwargs) -> Lowered: ...

class Stage:
    args_info: tp.Any
    @property
    def in_tree(self) -> jax.tree_util.PyTreeDef: ...
    @property
    def in_avals(self): ...
    @property
    def donate_argnums(self): ...

@dataclasses.dataclass(frozen=True, slots=True)
class Compiled(Stage):
    compiled: jax.stages.Compiled
    jit_wrapped: JitWrapped
    @property
    def args_info(self) -> tp.Any: ...
    @staticmethod
    def call(*args, **kwargs) -> None: ...
    def __call__(self, *args, **kwargs): ...
    @property
    def out_tree(self) -> jax.tree_util.PyTreeDef: ...
    def as_text(self) -> str | None: ...
    def cost_analysis(self) -> tp.Any | None: ...
    def memory_analysis(self) -> tp.Any | None: ...
    def runtime_executable(self) -> tp.Any | None: ...
    @property
    def input_shardings(self): ...
    @property
    def output_shardings(self): ...
    @property
    def input_layouts(self): ...

@dataclasses.dataclass(frozen=True, slots=True)
class Lowered(Stage):
    lowered: jax.stages.Lowered
    jit_wrapped: JitWrapped
    @property
    def args_info(self) -> tp.Any: ...
    @property
    def out_tree(self): ...
    @classmethod
    def from_flat_info(
        cls,
        lowering: tp.Any,
        in_tree: jax.tree_util.PyTreeDef,
        in_avals,
        donate_argnums: tuple[int, ...],
        out_tree: jax.tree_util.PyTreeDef,
        no_kwargs: bool = False,
    ): ...
    def compile(
        self,
        compiler_options: jax.stages.CompilerOptions | None = None,
    ) -> Compiled: ...
    def as_text(
        self,
        dialect: str | None = None,
        *,
        debug_info: bool = False,
    ) -> str: ...
    def compiler_ir(self, dialect: str | None = None) -> tp.Any | None: ...
    def cost_analysis(self) -> tp.Any | None: ...

@dataclasses.dataclass(frozen=True, slots=True)
class Traced(Stage):
    traced: jax.stages.Traced
    jit_wrapped: JitWrapped
    @property
    def out_info(self): ...
    def lower(
        self,
        *,
        lowering_platforms: tuple[str, ...] | None = None,
    ) -> Lowered: ...

@dataclasses.dataclass(frozen=True, slots=True)
class TreeCompiled(Stage):
    compiled: jax.stages.Compiled
    jit_wrapped: TreeJitWrapped
    @property
    def args_info(self) -> tp.Any: ...
    @staticmethod
    def call(*args, **kwargs) -> None: ...
    def __call__(self, *args, **kwargs): ...
    @property
    def out_tree(self) -> jax.tree_util.PyTreeDef: ...
    def as_text(self) -> str | None: ...
    def cost_analysis(self) -> tp.Any | None: ...
    def memory_analysis(self) -> tp.Any | None: ...
    def runtime_executable(self) -> tp.Any | None: ...
    @property
    def input_shardings(self): ...
    @property
    def output_shardings(self): ...
    @property
    def input_layouts(self): ...

@dataclasses.dataclass(frozen=True, slots=True)
class TreeLowered(Stage):
    lowered: jax.stages.Lowered
    jit_wrapped: TreeJitWrapped
    @property
    def args_info(self) -> tp.Any: ...
    @property
    def out_tree(self): ...
    def compile(
        self,
        compiler_options: jax.stages.CompilerOptions | None = None,
    ) -> TreeCompiled: ...
    def as_text(
        self,
        dialect: str | None = None,
        *,
        debug_info: bool = False,
    ) -> str: ...
    def compiler_ir(self, dialect: str | None = None) -> tp.Any | None: ...
    def cost_analysis(self) -> tp.Any | None: ...

@dataclasses.dataclass(frozen=True, slots=True)
class TreeTraced(Stage):
    traced: jax.stages.Traced
    jit_wrapped: TreeJitWrapped
    @property
    def out_info(self): ...
    def lower(
        self,
        *,
        lowering_platforms: tuple[str, ...] | None = None,
    ) -> TreeLowered: ...

@dataclasses.dataclass(eq=False)
class TreeShardMapFn:
    f: tp.Callable[..., tp.Any]
    def __post_init__(self) -> None: ...
    @extract.treemap_copy_args
    def __call__(self, *args): ...

@dataclasses.dataclass(eq=False)
class ShardMapFn:
    f: tp.Callable[..., tp.Any]
    in_specs: tp.Any
    out_specs: tp.Any
    kwarg_specs: tp.Any
    ctxtag: tp.Hashable
    def __post_init__(self) -> None: ...
    def __call__(self, *pure_args, **pure_kwargs): ...

@tp.overload
def shard_map(
    f: F,
    *,
    mesh: Mesh | AbstractMesh,
    in_specs: Specs,
    out_specs: Specs,
    axis_names: tp.AbstractSet[AxisName] = ...,
    check_vma: bool = True,
    graph: bool | None = None,
) -> F: ...
@tp.overload
def shard_map(
    *,
    mesh: Mesh | AbstractMesh,
    in_specs: Specs,
    out_specs: Specs,
    axis_names: tp.AbstractSet[AxisName] = ...,
    check_vma: bool = True,
    graph: bool | None = None,
) -> tp.Callable[[F], F]: ...
