from collections.abc import (
    Callable as Callable,
    Iterable,
)
from typing import Any, TypeVar

from _typeshed import Incomplete
from jax._src import tree_util as tree_util

T = TypeVar("T")

def all(tree: Any, *, is_leaf: Callable[[Any], bool] | None = None) -> bool: ...
def flatten(
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> tuple[list[tree_util.Leaf], tree_util.PyTreeDef]: ...
def leaves(
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> list[tree_util.Leaf]: ...
def map(
    f: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any: ...
def reduce(
    function: Callable[[T, Any], T],
    tree: Any,
    initializer: T | tree_util.Unspecified = ...,
    is_leaf: Callable[[Any], bool] | None = None,
) -> T: ...
def reduce_associative(
    operation: Callable[[T, T], T],
    tree: Any,
    *,
    identity: T | tree_util.Unspecified = ...,
    is_leaf: Callable[[Any], bool] | None = None,
) -> T: ...
def structure(
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> tree_util.PyTreeDef: ...
def transpose(
    outer_treedef: tree_util.PyTreeDef,
    inner_treedef: tree_util.PyTreeDef | None,
    pytree_to_transpose: Any,
) -> Any: ...
def unflatten(
    treedef: tree_util.PyTreeDef,
    leaves: Iterable[tree_util.Leaf],
) -> Any: ...
def flatten_with_path(
    tree: Any,
    is_leaf: Callable[..., bool] | None = None,
    is_leaf_takes_path: bool = False,
) -> tuple[list[tuple[tree_util.KeyPath, Any]], tree_util.PyTreeDef]: ...
def leaves_with_path(
    tree: Any,
    is_leaf: Callable[..., bool] | None = None,
    is_leaf_takes_path: bool = False,
) -> list[tuple[tree_util.KeyPath, Any]]: ...
def map_with_path(
    f: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[..., bool] | None = None,
    is_leaf_takes_path: bool = False,
) -> Any: ...
def broadcast(
    prefix_tree: Any,
    full_tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any: ...

static: Incomplete
