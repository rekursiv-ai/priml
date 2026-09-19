from collections.abc import Generator

import contextlib
import dataclasses
import functools
import threading
import typing as tp
import typing_extensions as tpe

from _typeshed import Incomplete
from flax import config as config
from flax.nnx import (
    filterlib as filterlib,
    reprlib as reprlib,
    statelib as statelib,
    traversals as traversals,
    variablelib as variablelib,
)
from flax.nnx.proxy_caller import (
    ApplyCaller as ApplyCaller,
    CallableProxy as CallableProxy,
    DelayedAccessor as DelayedAccessor,
)
from flax.nnx.statelib import (
    FlatState as FlatState,
    State as State,
    map_state as map_state,
)
from flax.nnx.variablelib import (
    V as V,
    Variable as Variable,
    is_array_ref as is_array_ref,
)
from flax.typing import (
    BaseConfigContext as BaseConfigContext,
    HashableMapping as HashableMapping,
    Key as Key,
    PathParts as PathParts,
    is_key_like as is_key_like,
)

import jax

A = tp.TypeVar("A")
B = tp.TypeVar("B")
C = tp.TypeVar("C")
F = tp.TypeVar("F", bound=tp.Callable)
HA = tp.TypeVar("HA", bound=tp.Hashable)
HB = tp.TypeVar("HB", bound=tp.Hashable)
KeyT = tp.TypeVar("KeyT", bound=Key)
Index = int
type Names = tp.Sequence[int]
Node = tp.TypeVar("Node")
Leaf = tp.TypeVar("Leaf")
AuxData = tp.TypeVar("AuxData")

@dataclasses.dataclass(frozen=True, slots=True)
class NoUpdate: ...

NO_UPDATE: Incomplete

@dataclasses.dataclass(frozen=True, slots=True)
class Repeated: ...

REPEATED: Incomplete

@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class ArrayRefOutput(reprlib.Representable):
    value: jax.Array
    def __nnx_repr__(self) -> Generator[Incomplete]: ...
    def __treescope_repr__(self, path, subtree_renderer): ...

LeafType: Incomplete
type GraphState = State[Key, LeafType]
type GraphFlatState = FlatState[LeafType]

def is_node_leaf(x: tp.Any) -> tpe.TypeGuard[LeafType]: ...

class IndexMap(dict[Index, tp.Any]):
    @staticmethod
    def from_refmap(refmap: RefMap) -> IndexMap: ...

class RefMap(tp.MutableMapping[tp.Any, int], reprlib.MappingReprMixin):
    def __init__(
        self,
        mapping: tp.Mapping[tp.Any, int]
        | tp.Iterable[tuple[tp.Any, int]]
        | None = None,
        /,
    ) -> None: ...
    @staticmethod
    def from_indexmap(indexmap: IndexMap) -> RefMap: ...
    def get(self, key: tp.Any, default: int | None = None) -> int | None: ...
    def __getitem__(self, key: tp.Any) -> int: ...
    def __setitem__(self, key: tp.Any, value: int): ...
    def __delitem__(self, key: tp.Any): ...
    def __len__(self) -> int: ...
    def __contains__(self, key: tp.Any) -> bool: ...
    def __iter__(self) -> tp.Iterator[tp.Any]: ...
    def items(self) -> tp.ItemsView[tp.Any, int]: ...

PythonRefMap = RefMap

@dataclasses.dataclass(frozen=True, slots=True)
class NodeImplBase(tp.Generic[Node, Leaf, AuxData]):
    type: type[Node]
    flatten: tp.Callable[[Node], tuple[tp.Sequence[tuple[Key, Leaf]], AuxData]]
    def node_dict(self, node: Node) -> dict[Key, tp.Any]: ...

@dataclasses.dataclass(frozen=True, slots=True)
class GraphNodeImpl(NodeImplBase[Node, Leaf, AuxData]):
    set_key: tp.Callable[[Node, Key, Leaf], None]
    pop_key: tp.Callable[[Node, Key], Leaf]
    create_empty: tp.Callable[[AuxData], Node]
    clear: tp.Callable[[Node], None]
    init: tp.Callable[[Node, tp.Iterable[tuple[Key, Leaf]]], None]

@dataclasses.dataclass(frozen=True, slots=True)
class PytreeNodeImpl(NodeImplBase[Node, Leaf, AuxData]):
    unflatten: tp.Callable[[tp.Sequence[tuple[Key, Leaf]], AuxData], Node]
    set_key: tp.Callable[[Node, Key, Leaf], None] | None
    pop_key: tp.Callable[[Node, Key], Leaf] | None

type NodeImpl[Node, Leaf, AuxData] = (
    GraphNodeImpl[Node, Leaf, AuxData] | PytreeNodeImpl[Node, Leaf, AuxData]
)
GRAPH_REGISTRY: dict[type, NodeImpl[tp.Any, tp.Any, tp.Any]]
PYTREE_REGISTRY: dict[type, PytreeNodeImpl[tp.Any, tp.Any, tp.Any]]

def register_graph_node_type(
    type: type,
    flatten: tp.Callable[[Node], tuple[tp.Sequence[tuple[Key, Leaf]], AuxData]],
    set_key: tp.Callable[[Node, Key, Leaf], None],
    pop_key: tp.Callable[[Node, Key], Leaf],
    create_empty: tp.Callable[[AuxData], Node],
    clear: tp.Callable[[Node], None],
    init: tp.Callable[[Node, tp.Iterable[tuple[Key, Leaf]]], None],
): ...
def register_pytree_node_type(
    type: type,
    flatten: tp.Callable[[Node], tuple[tp.Sequence[tuple[Key, Leaf]], AuxData]],
    unflatten: tp.Callable[[tp.Sequence[tuple[Key, Leaf]], AuxData], Node],
    *,
    set_key: tp.Callable[[Node, Key, Leaf], None] | None = None,
    pop_key: tp.Callable[[Node, Key], Leaf] | None = None,
): ...
def is_node(x: tp.Any) -> bool: ...
def is_graph_node(x: tp.Any) -> bool: ...
def is_node_type(x: type[tp.Any]) -> bool: ...
def get_node_impl(x: Node) -> NodeImpl[Node, tp.Any, tp.Any] | None: ...
def get_node_impl_for_type(x: type[Node]) -> NodeImpl[Node, tp.Any, tp.Any] | None: ...

@dataclasses.dataclass(frozen=True, repr=False)
class NodeRef(reprlib.Representable, tp.Generic[Node]):
    index: int
    def __nnx_repr__(self) -> Generator[Incomplete]: ...
    def __treescope_repr__(self, path, subtree_renderer): ...

@dataclasses.dataclass(frozen=True, repr=False)
class VariableDef(reprlib.Representable, tp.Generic[Node]):
    type: type[Node]
    index: int
    outer_index: int | None
    metadata: HashableMapping[str, tp.Any]
    array_refdef: ArrayRefDef | NodeRef | None
    def with_no_outer_index(self) -> VariableDef: ...
    def with_same_outer_index(self) -> VariableDef: ...
    def with_matching_outer_index(self, other) -> VariableDef: ...
    def __nnx_repr__(self) -> Generator[Incomplete]: ...
    def __treescope_repr__(self, path, subtree_renderer): ...

@dataclasses.dataclass(frozen=True, repr=False)
class ArrayRefDef(reprlib.Representable):
    index: int
    outer_index: int | None
    def with_no_outer_index(self): ...
    def with_same_outer_index(self): ...
    def with_matching_outer_index(self, other): ...
    def __nnx_repr__(self) -> Generator[Incomplete]: ...
    def __treescope_repr__(self, path, subtree_renderer): ...

@dataclasses.dataclass(frozen=True, repr=False, slots=True)
class NodeDef(reprlib.Representable, tp.Generic[Node]):
    type: type[Node]
    index: int | None
    outer_index: int | None
    num_attributes: int
    metadata: tp.Any
    def with_no_outer_index(self) -> NodeDef[Node]: ...
    def with_same_outer_index(self) -> NodeDef[Node]: ...
    def with_matching_outer_index(self, other) -> NodeDef[Node]: ...
    def __nnx_repr__(self) -> Generator[Incomplete]: ...
    def __treescope_repr__(self, path, subtree_renderer): ...

@dataclasses.dataclass(frozen=True, slots=True)
class TreeNodeDef(tp.Generic[Node]):
    type: type[Node]
    treedef: jax.tree_util.PyTreeDef
    path_index: tuple[tuple[PathParts, int], ...]
    def with_no_outer_index(self) -> TreeNodeDef[Node]: ...
    def with_same_outer_index(self) -> TreeNodeDef[Node]: ...
    def with_matching_outer_index(self, other) -> TreeNodeDef[Node]: ...

type NodeDefType[Node] = (
    NodeDef[Node] | NodeRef[Node] | VariableDef[Node] | ArrayRefDef | TreeNodeDef[Node]
)

@dataclasses.dataclass(frozen=True, slots=True)
class NodeAttr: ...

NODE_ATTR: Incomplete

@dataclasses.dataclass(frozen=True, slots=True)
class LeafAttr: ...

LEAF_ATTR: Incomplete
AttrType: Incomplete

@dataclasses.dataclass(frozen=True, slots=True)
class GraphDef(tp.Generic[Node]):
    nodes: list[NodeDefType[tp.Any]]
    attributes: list[tuple[Key, AttrType]]
    num_leaves: int
    def __hash__(self) -> int: ...
    def with_no_outer_index(self) -> GraphDef[Node]: ...
    def with_matching_outer_index(self, other) -> GraphDef[Node]: ...
    def with_same_outer_index(self) -> GraphDef[Node]: ...
    def apply(
        self,
        state: GraphState,
        *states: GraphState,
        graph: bool | None = None,
    ) -> ApplyCaller[tuple[GraphDef[Node], GraphState]]: ...

type PureState[Node] = tuple[GraphDef[Node], GraphState]

@tp.overload
def flatten(
    node: Node,
    /,
    *,
    ref_index: RefMap | None = ...,
    ref_outer_index: RefMap | None = ...,
    graph: bool = ...,
) -> tuple[GraphDef[Node], FlatState[tp.Any]]: ...
@tp.overload
def flatten(
    node: Node,
    /,
    *,
    with_paths: tp.Literal[True],
    ref_index: RefMap | None = ...,
    ref_outer_index: RefMap | None = ...,
    graph: bool = ...,
) -> tuple[GraphDef[Node], FlatState[tp.Any]]: ...
@tp.overload
def flatten(
    node: Node,
    /,
    *,
    with_paths: tp.Literal[False],
    ref_index: RefMap | None = ...,
    ref_outer_index: RefMap | None = ...,
    graph: bool = ...,
) -> tuple[GraphDef[Node], list[tp.Any]]: ...
@tp.overload
def flatten(
    node: Node,
    /,
    *,
    with_paths: bool,
    ref_index: RefMap | None = ...,
    ref_outer_index: RefMap | None = ...,
    graph: bool = ...,
) -> tuple[GraphDef[Node], FlatState[tp.Any] | list[tp.Any]]: ...

@dataclasses.dataclass(frozen=True, slots=True)
class DataElem:
    value: tp.Any

@dataclasses.dataclass(frozen=True, slots=True)
class StaticElem:
    value: tp.Any

def unflatten(
    graphdef: GraphDef[Node],
    state: State[Key, tp.Any] | FlatState[tp.Any] | list[tp.Any],
    /,
    *,
    index_ref: IndexMap | None = None,
    outer_index_outer_ref: IndexMap | None = None,
    copy_variables: bool = False,
) -> Node: ...
def graph_pop(
    node: tp.Any,
    filters: tuple[filterlib.Filter, ...],
) -> tuple[GraphState, ...]: ...

class StaticCache(tp.NamedTuple):
    graphdef: GraphDef[tp.Any]
    final_graphdef: GraphDef[tp.Any]
    paths: tuple[PathParts, ...]
    variables: list[Variable[tp.Any]]
    new_ref_index: RefMap
    new_index_ref: IndexMap
    @staticmethod
    def create(
        graphdef: GraphDef[tp.Any],
        paths: tuple[PathParts, ...],
        variables: list[Variable[tp.Any]],
        new_ref_index: RefMap,
    ): ...

@dataclasses.dataclass
class GraphContext(threading.local):
    update_context_stacks: dict[tp.Hashable, list[UpdateContext]] = dataclasses.field(
        default_factory=dict,
    )
    ref_index_stack: list[SplitContext] = dataclasses.field(default_factory=list)
    index_ref_stack: list[MergeContext] = dataclasses.field(default_factory=list)
    tmp_static_cache: tp.MutableMapping[tp.Any, StaticCache] | None = ...
    caching: bool = ...
    graph_mode_stack: list[bool] = dataclasses.field(default_factory=list)

GRAPH_CONTEXT: Incomplete

class set_graph_mode(BaseConfigContext):
    get_default: Incomplete
    get_stack: Incomplete

@contextlib.contextmanager
def static_cache(static_cache: tp.MutableMapping[tp.Any, StaticCache]): ...

cached_partial = functools.partial

@dataclasses.dataclass
class SplitContext:
    ctxtag: tp.Hashable | None
    ref_index: RefMap
    is_inner: bool | None
    @tp.overload
    def split(self, graph_node: A, /) -> tuple[GraphDef[A], GraphState]: ...
    @tp.overload
    def split(
        self,
        graph_node: A,
        first: filterlib.Filter,
        /,
    ) -> tuple[GraphDef[A], GraphState]: ...
    @tp.overload
    def split(
        self,
        graph_node: A,
        first: filterlib.Filter,
        second: filterlib.Filter,
        /,
        *filters: filterlib.Filter,
    ) -> tuple[GraphDef[A], GraphState, *tuple[GraphState, ...]]: ...
    @tp.overload
    def flatten(
        self,
        graph_node: A,
        /,
        *,
        with_paths: tp.Literal[False],
    ) -> tuple[GraphDef[A], list[tp.Any]]: ...
    @tp.overload
    def flatten(self, graph_node: A, /) -> tuple[GraphDef[A], FlatState[tp.Any]]: ...
    @tp.overload
    def flatten(
        self,
        graph_node: A,
        first: filterlib.Filter,
        /,
    ) -> tuple[GraphDef[A], FlatState[tp.Any]]: ...
    @tp.overload
    def flatten(
        self,
        graph_node: A,
        first: filterlib.Filter,
        second: filterlib.Filter,
        /,
        *filters: filterlib.Filter,
    ) -> tuple[GraphDef[A], FlatState[tp.Any], *tuple[FlatState[tp.Any], ...]]: ...

@contextlib.contextmanager
def split_context(ctxtag: tp.Hashable | None = None): ...

@dataclasses.dataclass
class MergeContext:
    ctxtag: tp.Hashable | None
    index_ref: IndexMap
    is_inner: bool | None
    def merge(
        self,
        graphdef: GraphDef[A],
        state: GraphState,
        /,
        *states: GraphState,
    ) -> A: ...
    def unflatten(
        self,
        graphdef: GraphDef[A],
        flat_state: GraphFlatState | list[tp.Any],
        /,
        *flat_states: GraphFlatState,
    ) -> A: ...

@tp.overload
@contextlib.contextmanager
def merge_context() -> tp.Generator[MergeContext]: ...
@tp.overload
@contextlib.contextmanager
def merge_context(
    ctxtag: tp.Hashable | None,
    inner: bool | None,
) -> tp.Generator[MergeContext]: ...

@dataclasses.dataclass
class UpdateContext:
    tag: tp.Hashable
    outer_ref_outer_index: RefMap | None
    outer_index_inner_ref: IndexMap | None
    outer_index_outer_ref: IndexMap | None
    inner_ref_outer_index: RefMap | None
    static_cache: tp.MutableMapping[tp.Any, StaticCache] | None
    def __hash__(self): ...
    def __eq__(self, other): ...
    def flatten_end(self, ref_index: RefMap): ...
    def unflatten_end(self, index_ref: IndexMap, inner_merge: bool): ...

@dataclasses.dataclass
class UpdateContextManager:
    tag: tp.Hashable
    def __enter__(self): ...
    def __exit__(self, *args) -> None: ...
    def __call__(self, f: F) -> F: ...

def update_context(tag: tp.Hashable): ...
def current_update_context(tag: tp.Hashable) -> UpdateContext: ...
@tp.overload
def split(
    graph_node: A,
    /,
    *,
    graph: bool | None = None,
) -> tuple[GraphDef[A], GraphState]: ...
@tp.overload
def split(
    graph_node: A,
    first: filterlib.Filter,
    /,
    *,
    graph: bool | None = None,
) -> tuple[GraphDef[A], GraphState]: ...
@tp.overload
def split(
    graph_node: A,
    first: filterlib.Filter,
    second: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
    graph: bool | None = None,
) -> tuple[GraphDef[A], GraphState, *tuple[GraphState, ...]]: ...
def merge(
    graphdef: GraphDef[A],
    state: tp.Any,
    /,
    *states: tp.Any,
    copy: bool = False,
) -> A: ...
def update(node, state: tp.Any, /, *states: tp.Any) -> None: ...
@tp.overload
def state(node, /, *, graph: bool | None = None) -> GraphState: ...
@tp.overload
def state(
    node,
    first: filterlib.Filter,
    /,
    *,
    graph: bool | None = None,
) -> GraphState: ...
@tp.overload
def state(
    node,
    first: filterlib.Filter,
    second: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
    graph: bool | None = None,
) -> tuple[GraphState, ...]: ...

variables = state

def graphdef(node: tp.Any, /, *, graph: bool | None = None) -> GraphDef[tp.Any]: ...
@tp.overload
def pop(node, filter: filterlib.Filter, /) -> GraphState: ...
@tp.overload
def pop(
    node,
    filter: filterlib.Filter,
    filter2: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
) -> tuple[GraphState, ...]: ...
def clone(node: Node, variables: bool = True, *, graph: bool | None = None) -> Node: ...
def vars_as(
    node: A,
    /,
    *,
    hijax: bool | None = None,
    ref: bool | None = None,
    mutable: bool | None = None,
    only: filterlib.Filter = ...,
    allow_duplicates: bool = False,
) -> A: ...
def pure(tree: A) -> A: ...
def call(
    graphdef_state: tuple[GraphDef[A], GraphState],
    /,
) -> ApplyCaller[tuple[GraphDef[A], GraphState]]: ...
def set_metadata(
    node: tp.Any,
    /,
    *,
    only: filterlib.Filter = ...,
    **metadata: tp.Any,
) -> None: ...
def iter_graph(
    node: tp.Any,
    /,
    *,
    graph: bool | None = None,
) -> tp.Iterator[tuple[PathParts, tp.Any]]: ...
def iter_children(
    node: tp.Any,
    /,
    *,
    graph: bool | None = None,
) -> tp.Iterator[tuple[Key, tp.Any]]: ...
def recursive_map(
    f: tp.Callable[[PathParts, tp.Any], tp.Any],
    node: tp.Any,
    /,
    *,
    graph: bool | None = None,
): ...
def find_duplicates(
    node: tp.Any,
    /,
    *,
    only: filterlib.Filter = ...,
) -> list[list[PathParts]]: ...

@dataclasses.dataclass(frozen=True, slots=True)
class Static(tp.Generic[A]):
    value: A

class GenericPytree: ...

def is_pytree_node(x: tp.Any, *, check_graph_registry: bool = True) -> bool: ...
def jax_to_nnx_path(jax_path: tuple, /): ...

class IndexesPytreeDef(tp.NamedTuple):
    key_index: HashableMapping[Key, int]
    treedef: jax.tree_util.PyTreeDef

PYTREE_NODE_IMPL: Incomplete
