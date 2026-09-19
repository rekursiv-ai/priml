from collections.abc import Callable as Callable
from typing import Any

import dataclasses

from _typeshed import Incomplete

@dataclasses.dataclass
class _ClassWrapperForGarbageCollection:
    obj: Any

@dataclasses.dataclass(frozen=True)
class _ObjectState:
    is_being_initialized: bool
    exc: Exception | None = ...
    obj: Any = ...

class _ObjectStore:
    def __init__(self) -> None: ...
    def get_or_create(self, uid: int, initializer: Callable[[], Any]) -> Any: ...
    def remove(self, uid: int) -> None: ...

SINGLETON_OBJECT_STORE: Incomplete
