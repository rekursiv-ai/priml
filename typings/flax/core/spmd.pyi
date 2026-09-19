import contextlib
import dataclasses
import threading

from flax.core import meta as meta
from flax.typing import (
    LogicalRules as LogicalRules,
    Sharding as Sharding,
)
from jax.sharding import PartitionSpec

import jax

def get_pspec(sharding, sharding_rules=None) -> PartitionSpec: ...
def shard_value(
    value,
    sharding,
    sharding_rules,
    mesh: jax.sharding.AbstractMesh | jax.sharding.Mesh | None,
): ...

@dataclasses.dataclass
class _AxisRules(threading.local):
    rules: LogicalRules = ...

def set_logical_axis_rules(rules: LogicalRules): ...
def get_logical_axis_rules() -> LogicalRules: ...
@contextlib.contextmanager
def logical_axis_rules(rules: LogicalRules): ...
def composite_rules(rule1, rule2): ...
def from_sharding_rules(
    sharding: Sharding,
    sharding_rules: LogicalRules,
) -> Sharding: ...
