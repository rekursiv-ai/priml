from collections.abc import Callable, Mapping, Sequence
from typing import Any, overload

from _typeshed import Incomplete
from flax import struct as struct

@struct.dataclass
class _EmptyNode: ...

empty_node: Incomplete
type IsLeafCallable = Callable[[tuple[Any, ...], Mapping[Any, Any]], bool]

@overload
def flatten_mapping(
    xs: Mapping[Any, Any],
    /,
    *,
    keep_empty_nodes: bool = False,
    is_leaf: IsLeafCallable | None = None,
    sep: None = None,
) -> dict[tuple[Any, ...], Any]: ...
@overload
def flatten_mapping(
    xs: Mapping[Any, Any],
    /,
    *,
    keep_empty_nodes: bool = False,
    is_leaf: IsLeafCallable | None = None,
    sep: str,
) -> dict[str, Any]: ...
def flatten_to_sequence(
    xs: Mapping[Any, Any],
    /,
    *,
    is_leaf: IsLeafCallable | None = None,
) -> list[tuple[Any, Any]]: ...
@overload
def unflatten_mapping(
    xs: Sequence[tuple[tuple[Any, ...], Any]],
    /,
    *,
    sep: None = None,
) -> dict[Any, Any]: ...
@overload
def unflatten_mapping(
    xs: Mapping[tuple[Any, ...], Any],
    /,
    *,
    sep: None = None,
) -> dict[Any, Any]: ...
@overload
def unflatten_mapping(xs: Mapping[str, Any], /, *, sep: str) -> dict[Any, Any]: ...
