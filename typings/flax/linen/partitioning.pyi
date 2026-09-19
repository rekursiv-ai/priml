from collections.abc import (
    Callable as Callable,
    Mapping,
)
from typing import Any

from _typeshed import Incomplete
from flax import (
    linen as nn,
    struct as struct,
)
from flax.core.frozen_dict import (
    freeze as freeze,
    unfreeze as unfreeze,
)
from flax.core.scope import (
    CollectionFilter as CollectionFilter,
    PRNGSequenceFilter as PRNGSequenceFilter,
)
from flax.linen.spmd import (
    RulesFallback as RulesFallback,
    logical_to_mesh as logical_to_mesh,
    logical_to_mesh_axes as logical_to_mesh_axes,
)
from flax.traverse_util import (
    flatten_dict as flatten_dict,
    unflatten_dict as unflatten_dict,
)
from flax.typing import (
    Array as Array,
    ArrayPytree as ArrayPytree,
    InOutAxis as InOutAxis,
    InOutScanAxis as InOutScanAxis,
    LogicalPartitionSpec as LogicalPartitionSpec,
    LogicalPartitionSpecPytree as LogicalPartitionSpecPytree,
    LogicalRules as LogicalRules,
    PartitionSpecPytree as PartitionSpecPytree,
)

import flax

@struct.dataclass
class AxisMetadata:
    names: LogicalPartitionSpecPytree = struct.field(pytree_node=False)

def param_with_axes(
    name: str,
    init_fn,
    *init_args,
    axes: tuple[str, ...] | None = None,
    module: nn.Module | None = None,
    **init_kwargs,
): ...

class PartitionedVariable(flax.core.scope.Variable):
    scope: Incomplete
    collection: Incomplete
    name: Incomplete
    axes: Incomplete
    fallback: Incomplete
    def __init__(
        self,
        scope,
        collection: str,
        name: str,
        axes: tuple[str, ...] | None = None,
        fallback: RulesFallback = ...,
    ) -> None: ...
    @property
    def value(self): ...
    @value.setter
    def value(self, value) -> None: ...

def variable_with_axes(
    collection: str,
    name: str,
    init_fn,
    *init_args,
    axes: tuple[str, ...] | None = None,
    module: nn.Module | None = None,
    fallback: RulesFallback = ...,
    **init_kwargs,
): ...
def get_axis_names(axes_metadata): ...
def scan_with_axes(
    target: flax.linen.transforms.Target,
    variable_axes: Mapping[CollectionFilter, InOutScanAxis] = {},
    variable_broadcast: CollectionFilter = False,
    variable_carry: CollectionFilter = False,
    split_rngs: Mapping[PRNGSequenceFilter, bool] = {},
    in_axes: int = 0,
    out_axes: int = 0,
    length: int | None = None,
    reverse: bool = False,
    unroll: int = 1,
    axis_name: str = "layers",
    axes_collections: tuple[str, ...] = ("params",),
    data_transform: Callable[..., Any] | None = None,
    methods=None,
) -> flax.linen.transforms.Target: ...
def vmap_with_axes(
    target: flax.linen.transforms.Target,
    variable_axes: Mapping[CollectionFilter, InOutAxis],
    split_rngs: Mapping[PRNGSequenceFilter, bool] = {},
    in_axes: int = 0,
    out_axes: int = 0,
    axis_size: int | None = None,
    axis_name: str | None = None,
    partitioning_axis_names: Mapping[Any, str] = {},
    spmd_axis_name: str | None = None,
    methods=None,
) -> flax.linen.transforms.Target: ...
def core_remat_static(
    fn,
    variables: bool = True,
    rngs: bool = True,
    concrete: bool = False,
    prevent_cse: bool = True,
    static_argnums=(),
    policy=None,
): ...
def remat(
    target,
    variables: bool = True,
    rngs: bool = True,
    concrete: bool = False,
    prevent_cse: bool = True,
    static_argnums=(),
    policy=None,
    methods=None,
): ...
