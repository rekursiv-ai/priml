from collections.abc import (
    Callable as Callable,
    Hashable,
    Iterable,
    Sequence,
)
from functools import cached_property as cached_property
from typing import Any, NamedTuple, TypeVar

import dataclasses
import functools

from _typeshed import Incomplete
from jax._src import traceback_util as traceback_util
from jax._src.lib import pytree as pytree
from jax._src.util import (
    safe_zip as safe_zip,
    set_module as set_module,
    unzip2 as unzip2,
)

export: Incomplete
T = TypeVar("T")
Typ = TypeVar("Typ", bound=type[Any])
H = TypeVar("H", bound=Hashable)
type Leaf = Any
type PyTree = Any
PyTreeDef: Incomplete
default_registry: Incomplete
none_leaf_registry: Incomplete
dispatch_registry: Incomplete

@export
def tree_flatten(
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> tuple[list[Leaf], PyTreeDef]: ...
@export
def tree_unflatten(treedef: PyTreeDef, leaves: Iterable[Leaf]) -> Any: ...
@export
def tree_leaves(
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> list[Leaf]: ...
@export
def tree_leaves_checked(treedef_expected: PyTreeDef, tree: Any) -> list[Leaf]: ...
@export
def tree_structure(
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTreeDef: ...
@export
def treedef_tuple(treedefs: Iterable[PyTreeDef]) -> PyTreeDef: ...
@export
def treedef_children(treedef: PyTreeDef) -> list[PyTreeDef]: ...
@export
def treedef_is_leaf(treedef: PyTreeDef) -> bool: ...
def treedef_is_strict_leaf(treedef: PyTreeDef) -> bool: ...
@export
def all_leaves(
    iterable: Iterable[Any],
    is_leaf: Callable[[Any], bool] | None = None,
) -> bool: ...

KeyEntry = TypeVar("KeyEntry", bound=Any)
type KeyLeafPair[KeyEntry: Any] = tuple[KeyEntry, Any]
type KeyLeafPairs = Iterable[KeyLeafPair]
type KeyPath[KeyEntry: Any] = tuple[KeyEntry, ...]

@export
def register_pytree_node(
    nodetype: type[T],
    flatten_func: Callable[[T], tuple[_Children, _AuxData]],
    unflatten_func: Callable[[_AuxData, _Children], T],
    flatten_with_keys_func: Callable[[T], tuple[KeyLeafPairs, _AuxData]] | None = None,
) -> None: ...
@export
def register_pytree_node_class(cls) -> Typ: ...
@export
def tree_map(
    f: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any: ...
@export
def build_tree(treedef: PyTreeDef, xs: Any) -> Any: ...
@export
def tree_transpose(
    outer_treedef: PyTreeDef,
    inner_treedef: PyTreeDef | None,
    pytree_to_transpose: Any,
) -> Any: ...

class _RegistryEntry(NamedTuple):
    to_iter: Incomplete
    from_iter: Incomplete

class Unspecified: ...

@export
def tree_reduce(
    function: Callable[[T, Any], T],
    tree: Any,
    initializer: T | Unspecified = ...,
    is_leaf: Callable[[Any], bool] | None = None,
) -> T: ...
@export
def tree_reduce_associative(
    operation: Callable[[T, T], T],
    tree: Any,
    *,
    identity: T | Unspecified = ...,
    is_leaf: Callable[[Any], bool] | None = None,
) -> T: ...
@export
def tree_all(tree: Any, *, is_leaf: Callable[[Any], bool] | None = None) -> bool: ...

class _HashableCallableShim:
    fun: Incomplete
    def __init__(self, fun) -> None: ...
    def __call__(self, *args, **kw): ...
    def __hash__(self): ...
    def __eq__(self, other): ...

class Partial(functools.partial):
    def __new__(klass, func, *args, **kw): ...

@export
def tree_broadcast(
    prefix_tree: Any,
    full_tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any: ...
def broadcast_prefix(
    prefix_tree: Any,
    full_tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> list[Any]: ...
def broadcast_flattened_prefix_with_treedef(
    prefix_leaves: list[Any],
    prefix_treedef: PyTreeDef,
    full_treedef: PyTreeDef,
) -> list[Any]: ...
def flatten_one_level(tree: Any) -> tuple[Iterable[Any], Hashable]: ...
def flatten_one_level_with_keys(
    tree: Any,
) -> tuple[Iterable[KeyLeafPair], Hashable]: ...
def prefix_errors(
    prefix_tree: Any,
    full_tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> list[Callable[[str], ValueError]]: ...
def equality_errors(
    tree1: Any,
    tree2: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Iterable[tuple[KeyPath, str, str, str]]: ...
def equality_errors_pytreedef(
    tree1: PyTreeDef,
    tree2: PyTreeDef,
) -> Iterable[tuple[KeyPath, str, str, str]]: ...

SequenceKey: Any
DictKey: Any
GetAttrKey: Any
FlattenedIndexKey: Any

@export
def keystr(keys: KeyPath, *, simple: bool = False, separator: str = "") -> str: ...
@export
def register_pytree_with_keys(
    nodetype: type[T],
    flatten_with_keys: Callable[[T], tuple[Iterable[KeyLeafPair], _AuxData]],
    unflatten_func: Callable[[_AuxData, Iterable[Any]], T],
    flatten_func: Callable[[T], tuple[Iterable[Any], _AuxData]] | None = None,
): ...
@export
def register_pytree_with_keys_class(cls) -> Typ: ...
@export
def register_dataclass(
    nodetype: Typ,
    data_fields: Sequence[str] | None = None,
    meta_fields: Sequence[str] | None = None,
    drop_fields: Sequence[str] = (),
) -> Typ: ...
@export
def register_static(cls) -> type[H]: ...
@export
def tree_flatten_with_path(
    tree: Any,
    is_leaf: Callable[..., bool] | None = None,
    is_leaf_takes_path: bool = False,
) -> tuple[list[tuple[KeyPath, Any]], PyTreeDef]: ...
@export
def tree_leaves_with_path(
    tree: Any,
    is_leaf: Callable[..., bool] | None = None,
    is_leaf_takes_path: bool = False,
) -> list[tuple[KeyPath, Any]]: ...

generate_key_paths = tree_leaves_with_path

@export
def tree_map_with_path(
    f: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[..., bool] | None = None,
    is_leaf_takes_path: bool = False,
) -> Any: ...

class FlatTree:
    vals: Incomplete
    tree: Incomplete
    statics: Incomplete
    def __init__(self, vals, treedef: PyTreeDef, statics) -> None: ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def map(self, f: Callable) -> FlatTree: ...
    def map2(self, f: Callable, t2: FlatTree) -> FlatTree: ...
    def map3(self, f: Callable, t2: FlatTree, t3: FlatTree) -> FlatTree: ...
    def unzip2(self) -> tuple[FlatTree, FlatTree]: ...
    @staticmethod
    def pack(tree): ...
    def unpack(self) -> tuple[FlatTree, ...]: ...
    @staticmethod
    def flatten(tree: PyTree) -> FlatTree: ...
    @staticmethod
    def flatten_static_argnums(args, static_argnums): ...
    @staticmethod
    def flatten_static_argnames(kwargs, static_argnames): ...
    @staticmethod
    def flatten_static_argnums_argnames(
        args,
        kwargs,
        static_argnums,
        static_argnames,
    ): ...
    def unflatten(self) -> PyTree: ...
    @property
    def tree_without_statics(self): ...
    def update(self, new_vals) -> FlatTree: ...
    @cached_property
    def paths(self) -> FlatTree: ...
    def __len__(self) -> int: ...
    @cached_property
    def len(self): ...
    def __iter__(self): ...

def unwrap_statics(pytree, statics): ...
def filter_statics_from_treedef(registry, treedef, statics): ...

@dataclasses.dataclass(frozen=True)
class Static:
    val: Any
    def __eq__(self, other): ...
