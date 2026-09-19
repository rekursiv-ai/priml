from collections.abc import Callable as Callable

from _typeshed import Incomplete
from jax._src import (
    api_util as api_util,
    config as config,
    tree_util as tree_util,
)
from jax._src.traceback_util import api_boundary as api_boundary
from jax._src.util import wraps as wraps
from jax.experimental.colocated_python import (
    func as func,
    obj_backend as obj_backend,
)

import jax

class _InstanceRegistry:
    def __init__(self) -> None: ...
    def new_instance(self) -> int: ...
    def update_devices(self, uid: int, device_set: set[jax.Device]) -> None: ...
    def pop_instance(self, uid: int) -> set[jax.Device]: ...

SINGLETON_INSTANCE_REGISTRY: Incomplete

def wrap_class(cls, cls_sourceinfo: str | None) -> type[object]: ...
