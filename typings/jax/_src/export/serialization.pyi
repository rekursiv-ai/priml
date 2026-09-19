from collections.abc import Callable as Callable
from typing import Any, TypeVar

from jax._src import (
    core as core,
    dtypes as dtypes,
    effects as effects,
    mesh as mesh,
    named_sharding as named_sharding,
    partition_spec as partition_spec,
    tree_util as tree_util,
)
from jax._src.export import (
    _export,
    shape_poly as shape_poly,
)
from jax._src.lib import xla_client as xla_client

T = TypeVar("T")
SerT = TypeVar("SerT")

def serialize(exp: _export.Exported, vjp_order: int = 0) -> bytearray: ...
def deserialize(ser: bytearray) -> _export.Exported: ...
def register_dtype_kind(dtype: Any, kind: int): ...
