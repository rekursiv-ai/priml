import dataclasses
import typing as tp

from _typeshed import Incomplete
from flax import (
    struct as struct,
    typing as typing,
)
from flax.core.frozen_dict import FrozenDict as FrozenDict
from flax.nnx import (
    extract as extract,
    filterlib as filterlib,
    graphlib as graphlib,
    spmd as spmd,
    statelib as statelib,
    variablelib as variablelib,
)
from flax.nnx.module import Module as Module
from flax.nnx.statelib import State as State
from flax.nnx.transforms.transforms import resolve_kwargs as resolve_kwargs
from flax.typing import (
    Leaf as Leaf,
    Missing as Missing,
    PytreeDeque as PytreeDeque,
)

import jax

A = tp.TypeVar("A")
C = tp.TypeVar("C")
B = tp.TypeVar("B")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
G = tp.TypeVar("G", bound=tp.Callable[..., tp.Any])
M = tp.TypeVar("M", bound=Module)
MA = tp.TypeVar("MA", bound=Module)
N = tp.TypeVar("N", bound=Module)
T = tp.TypeVar("T")
StrInt = tp.TypeVar("StrInt", str, int)
AxisName = tp.Hashable
type Leaves = list[Leaf]
Index = int

class Carry: ...

class StateAxes(extract.PrefixMapping, tp.Mapping):
    def __init__(
        self,
        filter_axes: statelib.State
        | tp.Mapping[filterlib.Filter, Index | type[Carry] | None]
        | tp.Iterable[tuple[filterlib.Filter, Index | type[Carry] | None]],
        /,
    ) -> None: ...
    @property
    def filters(self) -> tuple[filterlib.Filter, ...]: ...
    @property
    def axes(self) -> tuple[Index | type[Carry] | None, ...]: ...
    def map_prefix(
        self,
        path: typing.PathParts,
        variable: variablelib.Variable,
    ) -> tp.Any: ...
    def items(self): ...
    def __getitem__(self, key): ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def __eq__(self, other): ...
    def __hash__(self): ...

AxisFn: Incomplete

@dataclasses.dataclass(eq=False)
class TreeVmapFn:
    f: tp.Callable[..., tp.Any]
    def __post_init__(self) -> None: ...
    @extract.treemap_copy_args
    def __call__(self, *args, **kwargs): ...

@dataclasses.dataclass(eq=False)
class TreePmapFn:
    f: tp.Callable[..., tp.Any]
    def __post_init__(self) -> None: ...
    @extract.treemap_copy_args
    def __call__(self, *args, **kwargs): ...

@dataclasses.dataclass(eq=False)
class VmapFn:
    f: tp.Callable[..., tp.Any]
    transform_metadata: tp.Mapping[str, tp.Any]
    in_axes: tp.Any
    out_axes: tp.Any
    def __post_init__(self) -> None: ...
    def __call__(self, *pure_args: tuple[tp.Any, ...]): ...

@tp.overload
def vmap(
    *,
    in_axes: int | tp.Sequence[tp.Any] | None = 0,
    out_axes: tp.Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    transform_metadata: tp.Mapping[str, tp.Any] = ...,
    graph: bool | None = None,
) -> tp.Callable[[F], F]: ...
@tp.overload
def vmap(
    f: F,
    *,
    in_axes: int | tp.Sequence[tp.Any] | None = 0,
    out_axes: tp.Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    transform_metadata: tp.Mapping[str, tp.Any] = ...,
    graph: bool | None = None,
) -> F: ...

@dataclasses.dataclass(eq=False)
class PmapFn:
    f: tp.Callable[..., tp.Any]
    transform_metadata: tp.Mapping[str, tp.Any]
    in_axes: tp.Any
    out_axes: tp.Any
    def __post_init__(self) -> None: ...
    def __call__(self, *pure_args: tuple[tp.Any, ...]): ...

@tp.overload
def pmap(
    *,
    axis_name: AxisName | None = None,
    in_axes: tp.Any = 0,
    out_axes: tp.Any = 0,
    static_broadcasted_argnums: int | tp.Iterable[int] = (),
    devices: tp.Sequence[jax.Device] | None = None,
    backend: str | None = None,
    axis_size: int | None = None,
    donate_argnums: int | tp.Iterable[int] = (),
    global_arg_shapes: tuple[tuple[int, ...], ...] | None = None,
    transform_metadata: tp.Mapping[str, tp.Any] = ...,
    graph: bool | None = None,
) -> tp.Callable[[F], F]: ...
@tp.overload
def pmap(
    f: F,
    *,
    axis_name: AxisName | None = None,
    in_axes: tp.Any = 0,
    out_axes: tp.Any = 0,
    static_broadcasted_argnums: int | tp.Iterable[int] = (),
    devices: tp.Sequence[jax.Device] | None = None,
    backend: str | None = None,
    axis_size: int | None = None,
    donate_argnums: int | tp.Iterable[int] = (),
    global_arg_shapes: tuple[tuple[int, ...], ...] | None = None,
    transform_metadata: tp.Mapping[str, tp.Any] = ...,
    graph: bool | None = None,
) -> F: ...

class Broadcasted(struct.PyTreeNode):
    data: tp.Any

@dataclasses.dataclass(eq=False)
class ScanFn:
    f: tp.Callable[..., tp.Any]
    input_carry_argnum: int | tp.Literal["all"] | None
    output_carry_argnum: int | tp.Literal["all"] | None
    in_axes: tp.Any
    out_axes: tp.Any
    transform_metadata: tp.Mapping[str, tp.Any]
    def __post_init__(self) -> None: ...
    def __call__(
        self,
        carry: tuple[
            tp.Any,
            PytreeDeque[list[State]],
            PytreeDeque[list[State]],
            PytreeDeque[Broadcasted],
        ],
        scan_in: tuple[tp.Any, ...],
    ): ...

@dataclasses.dataclass(eq=False)
class TreeScanFn:
    f: tp.Callable[..., tp.Any]
    def __post_init__(self) -> None: ...
    @extract.treemap_copy_args
    def __call__(self, carry, x): ...

@tp.overload
def scan(
    *,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
    in_axes: int | type[Carry] | tuple[tp.Any, ...] | None = ...,
    out_axes: tp.Any = ...,
    transform_metadata: tp.Mapping[str, tp.Any] = ...,
    graph: bool | None = None,
) -> tp.Callable[[F], F]: ...
@tp.overload
def scan(
    f: F,
    *,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
    in_axes: int | type[Carry] | tuple[tp.Any, ...] | None = ...,
    out_axes: tp.Any = ...,
    transform_metadata: tp.Mapping[str, tp.Any] = ...,
    graph: bool | None = None,
) -> F: ...

@dataclasses.dataclass(eq=False)
class TreeWhileLoopBodyFn:
    f: tp.Callable[..., tp.Any]
    def __post_init__(self) -> None: ...
    @extract.treemap_copy_args
    def __call__(self, val): ...

@dataclasses.dataclass(eq=False)
class WhileLoopCondFn:
    f: tp.Callable[..., tp.Any]
    def __post_init__(self) -> None: ...
    def __call__(self, pure_val): ...

@dataclasses.dataclass(eq=False)
class WhileLoopBodyFn:
    f: tp.Callable[..., tp.Any]
    def __post_init__(self) -> None: ...
    def __call__(self, pure_val): ...

def while_loop(
    cond_fun: tp.Callable[[T], tp.Any],
    body_fun: tp.Callable[[T], T],
    init_val: T,
    *,
    graph: bool | None = None,
) -> T: ...

@dataclasses.dataclass(eq=False)
class TreeForiLoopBodyFn:
    f: tp.Callable[..., tp.Any]
    def __post_init__(self) -> None: ...
    @extract.treemap_copy_args
    def __call__(self, i, val): ...

@dataclasses.dataclass(eq=False)
class ForiLoopBodyFn:
    f: tp.Callable[..., tp.Any]
    def __post_init__(self) -> None: ...
    def __call__(self, i, pure_val_in): ...

def fori_loop(
    lower: int,
    upper: int,
    body_fun: tp.Callable[[int, T], T],
    init_val: T,
    *,
    unroll: int | bool | None = None,
    graph: bool | None = None,
) -> T: ...
