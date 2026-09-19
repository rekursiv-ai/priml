from collections.abc import Callable as Callable
from typing import Any

import threading

from _typeshed import Incomplete
from jax._src import (
    api as api,
    tree_util as tree_util,
)

DeviceList: Incomplete

class _CommonObjectState(threading.local):
    common_obj_index: dict[Any, int] | None
    common_obj: list[Any] | None
    def __init__(self) -> None: ...
