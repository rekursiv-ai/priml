from typing import Any

from _typeshed import Incomplete
from jax._src import (
    api as api,
    config as config,
    core as core,
    dtypes as dtypes,
    literals as literals,
    tree_util as tree_util,
    xla_bridge as xla_bridge,
)
from jax._src.lax import lax as lax
from jax._src.lib import xla_client as xc
from jax._src.numpy import util as util
from jax._src.sharding import Sharding as Sharding
from jax._src.sharding_impls import (
    NamedSharding as NamedSharding,
    PartitionSpec as P,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DTypeLike as DTypeLike,
)

logger: Incomplete
export: Incomplete
cuda_plugin_extension: Incomplete
rocm_plugin_extension: Incomplete
name: Incomplete
module_name: Incomplete

@export
def array(
    object: Any,
    dtype: DTypeLike | None = None,
    copy: bool = True,
    order: str | None = "K",
    ndmin: int = 0,
    *,
    device: xc.Device | Sharding | None = None,
    out_sharding: NamedSharding | P | None = None,
) -> Array: ...
@export
def asarray(
    a: Any,
    dtype: DTypeLike | None = None,
    order: str | None = None,
    *,
    copy: bool | None = None,
    device: xc.Device | Sharding | None = None,
    out_sharding: NamedSharding | P | None = None,
) -> Array: ...
