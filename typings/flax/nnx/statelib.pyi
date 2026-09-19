from collections.abc import Generator, MutableMapping

import typing as tp

from _typeshed import Incomplete
from flax.nnx import (
    filterlib as filterlib,
    reprlib as reprlib,
    traversals as traversals,
    variablelib as variablelib,
)
from flax.typing import (
    Key as Key,
    PathParts as PathParts,
)

A = tp.TypeVar("A")
K = tp.TypeVar("K", bound=tp.Hashable)
S = tp.TypeVar("S", bound=State)
V = tp.TypeVar("V")
type ExtractValueFn = tp.Callable[[tp.Any], tp.Any]
type SetValueFn[V] = tp.Callable[[V, tp.Any], V]

class NestedStateRepr(reprlib.Representable):
    state: Incomplete
    def __init__(self, state: State) -> None: ...
    def __nnx_repr__(self) -> Generator[Incomplete]: ...
    def __treescope_repr__(self, path, subtree_renderer): ...

class FlatState(tp.Sequence[tuple[PathParts, V]], reprlib.Representable):
    def __init__(
        self,
        items: tp.Iterable[tuple[PathParts, V]],
        /,
        *,
        sort: bool,
    ) -> None: ...
    @staticmethod
    def from_sorted_keys_values(
        keys: tuple[PathParts, ...],
        values: list[V],
        /,
    ) -> FlatState[V]: ...
    @property
    def paths(self) -> tuple[PathParts, ...]: ...
    @property
    def leaves(self) -> list[V]: ...
    def __nnx_repr__(self) -> Generator[Incomplete]: ...
    @tp.overload
    def __getitem__(self, index: int) -> tuple[PathParts, V]: ...
    @tp.overload
    def __getitem__(self, index: slice) -> FlatState[V]: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> tp.Iterator[tuple[PathParts, V]]: ...
    def to_nested_state(self) -> State[Key, V]: ...
    @tp.overload
    def split(self, first: filterlib.Filter, /) -> FlatState[V]: ...
    @tp.overload
    def split(
        self,
        first: filterlib.Filter,
        second: filterlib.Filter,
        /,
        *filters: filterlib.Filter,
    ) -> tuple[FlatState[V], ...]: ...
    @tp.overload
    def split(
        self,
        /,
        *filters: filterlib.Filter,
    ) -> FlatState[V] | tuple[FlatState[V], ...]: ...
    @tp.overload
    def filter(self, first: filterlib.Filter, /) -> FlatState[V]: ...
    @tp.overload
    def filter(
        self,
        first: filterlib.Filter,
        second: filterlib.Filter,
        /,
        *filters: filterlib.Filter,
    ) -> tuple[FlatState[V], ...]: ...
    @staticmethod
    def merge(
        flat_state: tp.Iterable[tuple[PathParts, V]],
        /,
        *flat_states: tp.Iterable[tuple[PathParts, V]],
    ) -> FlatState[V]: ...

class State(MutableMapping[K, V], reprlib.Representable):
    def __init__(
        self,
        mapping: tp.Mapping[K, tp.Mapping | V] | tp.Iterator[tuple[K, tp.Mapping | V]],
        /,
        *,
        _copy: bool = True,
    ) -> None: ...
    @property
    def raw_mapping(self) -> dict[K, tp.Mapping[K, tp.Any] | V]: ...
    def __contains__(self, key) -> bool: ...
    def __getitem__(self, key: K) -> State | V: ...
    def __getattr__(self, key: K) -> State | V: ...
    def __setitem__(self, key: K, value: State | V) -> None: ...
    __setattr__ = __setitem__
    def __delitem__(self, key: K) -> None: ...
    def __iter__(self) -> tp.Iterator[K]: ...
    def __len__(self) -> int: ...
    def __nnx_repr__(self) -> Generator[Incomplete]: ...
    def __treescope_repr__(self, path, subtree_renderer): ...
    def map(self, f: tp.Callable[[tuple, V], V]) -> State[K, V]: ...
    def flat_state(self) -> FlatState[V]: ...
    @classmethod
    def from_flat_path(
        cls,
        flat_state: tp.Mapping[PathParts, V] | tp.Iterable[tuple[PathParts, V]],
        /,
    ): ...
    def to_pure_dict(
        self,
        extract_fn: ExtractValueFn | None = None,
    ) -> dict[str, tp.Any]: ...
    def replace_by_pure_dict(
        self,
        pure_dict: dict[str, tp.Any],
        replace_fn: SetValueFn | None = None,
    ): ...
    @tp.overload
    def split(self, first: filterlib.Filter, /) -> State[K, V]: ...
    @tp.overload
    def split(
        self,
        first: filterlib.Filter,
        second: filterlib.Filter,
        /,
        *filters: filterlib.Filter,
    ) -> tuple[State[K, V], ...]: ...
    @tp.overload
    def split(
        self,
        /,
        *filters: filterlib.Filter,
    ) -> State[K, V] | tuple[State[K, V], ...]: ...
    @tp.overload
    def filter(self, first: filterlib.Filter, /) -> State[K, V]: ...
    @tp.overload
    def filter(
        self,
        first: filterlib.Filter,
        second: filterlib.Filter,
        /,
        *filters: filterlib.Filter,
    ) -> tuple[State[K, V], ...]: ...
    @classmethod
    def merge(cls, state: tp.Mapping[K, V], /, *states: tp.Mapping[K, V]): ...
    def __or__(self, other: State[K, V]) -> State[K, V]: ...
    def __sub__(self, other: State[K, V]) -> State[K, V]: ...
    def __init_subclass__(cls) -> None: ...

def map_state(f: tp.Callable[[tuple, tp.Any], tp.Any], state: State) -> State: ...
def to_flat_state(state: State) -> FlatState: ...
def from_flat_state(
    flat_state: tp.Mapping[PathParts, V] | tp.Iterable[tuple[PathParts, V]],
    *,
    cls=...,
) -> State: ...
def to_pure_dict(
    state: State,
    extract_fn: ExtractValueFn | None = None,
) -> dict[str, tp.Any]: ...
def restore_int_paths(pure_dict: dict[str, tp.Any]): ...
def replace_by_pure_dict(
    state: State,
    pure_dict: dict[str, tp.Any],
    replace_fn: SetValueFn | None = None,
): ...
@tp.overload
def split_state(state: State, first: filterlib.Filter, /) -> State: ...
@tp.overload
def split_state(
    state: State,
    first: filterlib.Filter,
    second: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
) -> tuple[State, ...]: ...
@tp.overload
def split_state(
    state: State,
    /,
    *filters: filterlib.Filter,
) -> State | tuple[State, ...]: ...
@tp.overload
def filter_state(state: State, first: filterlib.Filter, /) -> State: ...
@tp.overload
def filter_state(
    state: State,
    first: filterlib.Filter,
    second: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
) -> tuple[State, ...]: ...
def merge_state(state: tp.Mapping, /, *states: tp.Mapping, cls=...) -> State: ...
def diff(state: State, other: State) -> State: ...
def create_path_filters(state: State): ...
