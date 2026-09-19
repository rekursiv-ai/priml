from collections import deque

import dataclasses
import typing as tp

from _typeshed import Incomplete
from flax import struct as struct
from flax.nnx import (
    extract as extract,
    filterlib as filterlib,
    graphlib as graphlib,
    variablelib as variablelib,
)
from flax.nnx.statelib import State as State
from flax.nnx.transforms import general as general
from flax.nnx.transforms.transforms import resolve_kwargs as resolve_kwargs
from flax.typing import (
    MISSING as MISSING,
    Missing as Missing,
)

A = tp.TypeVar("A")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
AxisName = tp.Hashable

@dataclasses.dataclass(frozen=True)
class DiffState:
    argnum: int
    filter: filterlib.Filter

@dataclasses.dataclass(eq=False)
class TreeGradFn:
    f: tp.Callable[..., tp.Any]
    has_aux: bool
    def __post_init__(self) -> None: ...
    @extract.treemap_copy_args
    def __call__(self, *args, **kwargs): ...

@dataclasses.dataclass(eq=False)
class GradFn:
    f: tp.Callable[..., tp.Any]
    has_aux: bool
    nondiff_states: deque[State | None]
    def __post_init__(self) -> None: ...
    def __call__(self, *pure_args): ...

@tp.overload
def grad(
    f: tp.Callable[..., tp.Any],
    *,
    argnums: int | DiffState | tp.Sequence[int | DiffState] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
    graph: bool | None = None,
) -> tp.Callable[..., tp.Any]: ...
@tp.overload
def grad(
    *,
    argnums: int | DiffState | tp.Sequence[int | DiffState] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
    graph: bool | None = None,
) -> tp.Callable[[tp.Callable[..., tp.Any]], tp.Callable[..., tp.Any]]: ...
@tp.overload
def value_and_grad(
    f: tp.Callable[..., tp.Any],
    *,
    argnums: int | DiffState | tp.Sequence[int | DiffState] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
    graph: bool | None = None,
) -> tp.Callable[..., tp.Any]: ...
@tp.overload
def value_and_grad(
    *,
    argnums: int | DiffState | tp.Sequence[int | DiffState] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
    graph: bool | None = None,
) -> tp.Callable[[tp.Callable[..., tp.Any]], tp.Callable[..., tp.Any]]: ...

@dataclasses.dataclass(eq=False)
class TreeVjpFn:
    f: tp.Callable[..., tp.Any]
    has_aux: bool
    def __post_init__(self) -> None: ...
    @extract.treemap_copy_args
    def __call__(self, *args): ...

@tp.overload
def vjp(
    f: tp.Callable[..., tp.Any],
    *primals: tp.Any,
    has_aux: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
    graph: bool | None = None,
) -> tuple[tp.Any, tp.Callable] | tuple[tp.Any, tp.Callable, tp.Any]: ...
@tp.overload
def vjp(
    *,
    has_aux: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
    graph: bool | None = None,
) -> tp.Callable[[tp.Callable[..., tp.Any]], tp.Callable[..., tp.Any]]: ...

@dataclasses.dataclass(eq=False)
class TreeJvpFn:
    f: tp.Callable[..., tp.Any]
    has_aux: bool
    def __post_init__(self) -> None: ...
    @extract.treemap_copy_args
    def __call__(self, *args): ...

@tp.overload
def jvp(
    f: tp.Callable[..., tp.Any],
    primals: tuple[tp.Any, ...],
    tangents: tuple[tp.Any, ...],
    *,
    has_aux: bool = False,
    graph: bool | None = None,
) -> tuple[tp.Any, ...]: ...
@tp.overload
def jvp(
    *,
    has_aux: bool = False,
    graph: bool | None = None,
) -> tp.Callable[[tp.Callable[..., tp.Any]], tp.Callable[..., tp.Any]]: ...
@tp.overload
def jvp(
    f: tp.Callable[..., tp.Any],
    *,
    has_aux: bool = False,
    graph: bool | None = None,
) -> tp.Callable[..., tp.Any]: ...

@dataclasses.dataclass(eq=False)
class TreeCustomVjpFn:
    f: tp.Callable[..., tp.Any]
    def __post_init__(self) -> None: ...
    @extract.treemap_copy_args
    def __call__(self, *args): ...

@dataclasses.dataclass(eq=False)
class TreeFwdFn:
    fwd: tp.Callable[..., tp.Any]
    def __post_init__(self) -> None: ...
    @extract.treemap_copy_args
    def __call__(self, *args): ...

@dataclasses.dataclass(eq=False)
class TreeBwdFn:
    bwd: tp.Callable[..., tp.Any]
    def __post_init__(self) -> None: ...
    @extract.treemap_copy_args
    def __call__(self, *args): ...

class TreeCustomVjp(tp.Generic[A]):
    fun: Incomplete
    nondiff_argnums: Incomplete
    custom_vjp_fn: Incomplete
    def __init__(
        self,
        fun: tp.Callable[..., A],
        nondiff_argnums: tuple[int, ...],
    ) -> None: ...
    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> A: ...
    fwd: Incomplete
    bwd: Incomplete
    symbolic_zeros: Incomplete
    def defvjp(
        self,
        fwd: tp.Callable[..., tuple[A, tp.Any]],
        bwd: tp.Callable[..., tuple[tp.Any, ...]],
        symbolic_zeros: bool = False,
    ) -> None: ...

@dataclasses.dataclass(eq=False)
class CustomVjpFnWrapper:
    f: tp.Callable[..., tp.Any]
    jax_nondiff_argnums: tuple[int, ...]
    ctxtag: str
    nondiff_states: list[extract.GraphDefState]
    nodedefs: deque[graphlib.GraphDef]
    def __post_init__(self) -> None: ...
    def __call__(self, *pure_args): ...

@dataclasses.dataclass(eq=False)
class FwdFn:
    fwd: tp.Callable[..., tp.Any]
    nondiff_argnums: tuple[int, ...]
    ctxtag: str
    nondiff_states: list[extract.GraphDefState]
    nodedefs: deque[graphlib.GraphDef]
    def __post_init__(self) -> None: ...
    def __call__(self, *pure_args): ...

@dataclasses.dataclass(eq=False)
class BwdFn:
    bwd: tp.Callable[..., tp.Any]
    tree_node_args: tuple[tp.Any, ...]
    def __post_init__(self) -> None: ...
    def __call__(self, *args): ...

class CustomVjp(tp.Generic[A]):
    jax_nondiff_argnums: Incomplete
    ctxtag: Incomplete
    fun: Incomplete
    fwd: tp.Callable | None
    bwd: tp.Callable | None
    symbolic_zeros: bool | None
    nondiff_argnums: Incomplete
    diff_filter: dict[int, tp.Literal[False] | DiffState]
    def __init__(
        self,
        fun: tp.Callable[..., A],
        nondiff_argnums: tuple[int | DiffState, ...],
    ) -> None: ...
    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> A: ...
    def defvjp(
        self,
        fwd: tp.Callable[..., tuple[A, tp.Any]],
        bwd: tp.Callable[..., tuple[tp.Any, ...]],
        symbolic_zeros: bool = False,
    ) -> None: ...

@tp.overload
def custom_vjp(
    fun: tp.Callable[..., A],
    *,
    nondiff_argnums: tuple[int | DiffState, ...] = (),
    graph: bool | None = None,
) -> CustomVjp[A] | TreeCustomVjp[A]: ...
@tp.overload
def custom_vjp(
    *,
    nondiff_argnums: tuple[int | DiffState, ...] = (),
    graph: bool | None = None,
) -> tp.Callable[[tp.Callable[..., A]], CustomVjp[A] | TreeCustomVjp[A]]: ...

@dataclasses.dataclass(eq=False)
class TreeRematFn:
    f: tp.Callable[..., tp.Any]
    def __post_init__(self) -> None: ...
    @extract.treemap_copy_args
    def __call__(self, *args, **kwargs): ...

@tp.overload
def remat(
    *,
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: tp.Callable[..., bool] | None = None,
    graph: bool | None = None,
) -> tp.Callable[[F], F]: ...
@tp.overload
def remat(
    f: F,
    *,
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: tp.Callable[..., bool] | None = None,
    graph: bool | None = None,
) -> F: ...
