from typing import Any

from _typeshed import Incomplete
from jax._src import (
    array as array,
    dtypes as dtypes,
    xla_bridge as xla_bridge,
)
from jax._src.api import device_put as device_put
from jax._src.lib import (
    _jax,
    xla_client as xla_client,
)
from jax._src.sharding import Sharding as Sharding
from jax._src.typing import (
    Array as Array,
    DLDeviceType as DLDeviceType,
    DTypeLike as DTypeLike,
)

import numpy as np

DLPACK_VERSION: Incomplete
MIN_DLPACK_VERSION: Incomplete
SUPPORTED_DTYPES: frozenset[DTypeLike]
SUPPORTED_DTYPES_SET: frozenset[np.dtype]

def is_supported_dtype(dtype: DTypeLike) -> bool: ...
def to_dlpack(
    x: Array,
    stream: int | Any | None = None,
    src_device: _jax.Device | None = None,
    dl_device: tuple[DLDeviceType, int] | None = None,
    max_version: tuple[int, int] | None = None,
    copy: bool | None = None,
): ...
def from_dlpack(
    external_array,
    device: _jax.Device | Sharding | None = None,
    copy: bool | None = None,
): ...
