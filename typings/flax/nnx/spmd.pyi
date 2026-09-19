import typing as tp

from flax.nnx import (
    graphlib as graphlib,
    variablelib as variablelib,
)
from flax.nnx.transforms.transforms import eval_shape as eval_shape
from flax.typing import Sharding as Sharding
from jax.sharding import PartitionSpec

import jax

A = tp.TypeVar("A")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
PARTITION_NAME: str

def add_axis(tree: A, index: int, transform_metadata: tp.Mapping) -> A: ...
def remove_axis(
    tree: A,
    index: int,
    transform_metadata: tp.Mapping[tp.Any, tp.Any],
) -> A: ...
def with_partitioning(
    initializer: F,
    sharding: Sharding,
    mesh: jax.sharding.Mesh | None = None,
    **metadata: tp.Any,
) -> F: ...
def get_var_pspec(v: variablelib.Variable) -> PartitionSpec | None: ...
def get_partition_spec(tree: A) -> A: ...
def get_named_sharding(tree: A, mesh: jax.sharding.Mesh) -> A: ...
def get_abstract_model(init_fn, mesh, *, graph: bool | None = None): ...
