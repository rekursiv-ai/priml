import abc
import typing as tp

from flax import (
    struct as struct,
    typing as typing,
)
from flax.nnx import (
    graphlib as graphlib,
    variablelib as variablelib,
)
from flax.nnx.pytreelib import Pytree as Pytree
from flax.typing import (
    Missing as Missing,
    PathParts as PathParts,
)

A = tp.TypeVar("A")
Index = int
KeyEntry = tp.TypeVar("KeyEntry", bound=tp.Hashable)
type KeyPath[KeyEntry: tp.Hashable] = tuple[KeyEntry, ...]
type Prefix = tp.Any
type Leaf = tp.Any

class PrefixMapping(abc.ABC, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def map_prefix(
        self,
        path: typing.PathParts,
        variable: variablelib.Variable,
        /,
    ) -> tp.Any: ...

def check_consistent_aliasing(
    node: tp.Any,
    prefix: tp.Any,
    /,
    *,
    node_prefixes: dict[int, list[tuple[PathParts, tp.Any]]] | None = None,
): ...
def broadcast_prefix(
    prefix_tree: tp.Any,
    full_tree: tp.Any,
    prefix_is_leaf: tp.Callable[[tp.Any], bool] | None = None,
    tree_is_leaf: tp.Callable[[tp.Any], bool] | None = None,
) -> list[tp.Any]: ...

class GraphDefState(struct.PyTreeNode):
    graphdef: graphlib.GraphDef[tp.Any] = struct.field(pytree_node=False)
    state: graphlib.GraphState = struct.field(pytree_node=True)

S = tp.TypeVar("S", bound=graphlib.GraphState | graphlib.GraphFlatState | list[tp.Any])

class NodeStates(struct.PyTreeNode):
    states: tuple[tp.Any, ...]
    metadata: tp.Any = struct.field(pytree_node=False)
    @property
    def graphdef(self) -> graphlib.GraphDef[tp.Any]: ...
    @property
    def state(self) -> tp.Any: ...
    @classmethod
    def from_split(
        cls,
        graphdef: graphlib.GraphDef[tp.Any] | None,
        state: tp.Any,
        /,
        *states: tp.Any,
        metadata: tp.Any = None,
    ): ...
    @classmethod
    def from_states(cls, state: tp.Any, *states: tp.Any): ...
    @classmethod
    def from_prefixes(
        cls,
        prefixes: tp.Iterable[tp.Any],
        /,
        *,
        metadata: tp.Any = None,
    ): ...

def default_split_fn(
    ctx: graphlib.SplitContext,
    path: KeyPath,
    prefix: Prefix,
    leaf: Leaf,
) -> tp.Any: ...
def to_tree(
    tree,
    /,
    *,
    prefix: tp.Any = ...,
    split_fn: tp.Callable[[graphlib.SplitContext, KeyPath, Prefix, Leaf], tp.Any] = ...,
    map_non_graph_nodes: bool = False,
    ctxtag: tp.Hashable | None = None,
    check_aliasing: bool = True,
) -> tp.Any: ...
def merge_tree_node(
    ctx: graphlib.MergeContext,
    path: KeyPath,
    prefix: Prefix,
    leaf: Leaf,
) -> tp.Any: ...
def is_tree_node(x): ...
def from_tree(
    tree: tp.Any,
    /,
    *,
    prefix: tp.Any = ...,
    merge_fn: tp.Callable[[graphlib.MergeContext, KeyPath, Prefix, Leaf], tp.Any] = ...,
    is_node_leaf: tp.Callable[[Leaf], bool] = ...,
    is_leaf: tp.Callable[[Leaf], bool] = ...,
    map_non_graph_nodes: bool = False,
    is_inner: bool | None = None,
    ctxtag: tp.Hashable | None = None,
) -> tp.Any: ...
def clear_non_graph_nodes(tree): ...
def updates_and_snapshot(args: A) -> tuple[A, A]: ...

class _InputsAndOutputs(tp.NamedTuple):
    args: tuple
    kwargs: dict | None
    output: tp.Any

@tp.overload
def check_no_aliases(args: tuple[tp.Any, ...], output: tp.Any) -> None: ...
@tp.overload
def check_no_aliases(
    args: tuple[tp.Any, ...],
    kwargs: dict[str, tp.Any],
    output: tp.Any,
) -> None: ...

type MaskFn = tp.Callable[[PathParts, variablelib.Variable, variablelib.Variable], bool]

def mask_variable_updates(
    current_tree: A,
    snapshot_tree: A,
    *,
    keep_fn: MaskFn | None = None,
) -> A: ...
def apply_variable_updates(args_tree: A, updates_tree: A) -> None: ...
def treemap_copy_args(f): ...
def check_same_variables(inputs, outputs, transform_name: str = ""): ...
def update_carry_variables(init_val, val_out): ...
