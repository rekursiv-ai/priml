from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

import enum

from flax import struct as struct
from flax.core import meta as meta
from flax.core.spmd import get_logical_axis_rules as get_logical_axis_rules
from flax.typing import (
    Array as Array,
    ArrayPytree as ArrayPytree,
    LogicalNames as LogicalNames,
    LogicalPartitionSpec as LogicalPartitionSpec,
    LogicalPartitionSpecPytree as LogicalPartitionSpecPytree,
    LogicalRules as LogicalRules,
)

import jax

class _UnassignedAxis:
    def __bool__(self) -> bool: ...

def logical_to_mesh_axes(
    array_dim_names: Sequence[str | None] | None,
    rules: LogicalRules | None = None,
) -> jax.sharding.PartitionSpec | None: ...
def logical_to_mesh(tree: Any, rules: LogicalRules | None = None) -> Any: ...
def logical_to_mesh_sharding(
    tree: Any,
    mesh: jax.sharding.Mesh,
    rules: LogicalRules | None = None,
) -> Any: ...

class RulesFallback(enum.Enum):
    AXIS_IS_UNSHARDED = "axis_is_unsharded"
    RAISE_ERROR = "raise_error"
    NO_CONSTRAINT = "no_constraint"

def with_logical_constraint(
    x: ArrayPytree,
    logical_axis_resources: LogicalPartitionSpecPytree,
    rules: LogicalRules | None = None,
    mesh: jax.sharding.Mesh | None = None,
    fallback: RulesFallback = ...,
): ...

class LogicallyPartitioned(meta.Partitioned):
    rules: LogicalRules | None = struct.field(default=None, pytree_node=False)
    def __eq__(self, other): ...
    def unbox(self, apply_constraint: bool = True) -> Any: ...
    def to_nnx_metadata(self) -> dict[str, Any]: ...
    @classmethod
    def from_nnx_metadata(cls, metadata: dict[str, Any]): ...

def with_logical_partitioning(
    fn: Callable[..., Any],
    names: LogicalNames,
    mesh: jax.sharding.Mesh | None = None,
    rules: LogicalRules | None = None,
) -> Callable[..., LogicallyPartitioned]: ...
