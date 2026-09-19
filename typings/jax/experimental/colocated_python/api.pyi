from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any, overload

from jax._src import (
    api_util as api_util,
    util as util,
)
from jax.experimental.colocated_python.func import make_callable as make_callable
from jax.experimental.colocated_python.obj import wrap_class as wrap_class

import jax

@overload
def colocated_cpu_devices(
    devices_or_mesh: Sequence[jax.Device],
) -> Sequence[jax.Device]: ...
@overload
def colocated_cpu_devices(devices_or_mesh: jax.sharding.Mesh) -> jax.sharding.Mesh: ...
def colocated_python(fun: Callable[..., Any]): ...
def colocated_python_class(cls) -> type[object]: ...
