from collections.abc import Generator
from concurrent.futures import Future
from types import ModuleType
from typing import Any, TypeVar

from _typeshed import Incomplete
from jax.export import (
    register_pytree_node_serialization as register_pytree_node_serialization,
)

__all__ = [
    "deserialize_pytreedef",
    "register_pytree_node_serialization",
    "serialize_pytreedef",
]

T = TypeVar("T")
PickleModule = ModuleType

class PyTreeFuture(Future[Any]):
    def __init__(self, future: Future[Any]) -> None: ...
    def done(self): ...
    def result(self, *args, **kw): ...
    def __await__(self) -> Generator[None, None, Incomplete]: ...

def serialize_pytreedef(node) -> dict[str, Any]: ...
def deserialize_pytreedef(pytreedef_repr: dict[str, Any]): ...
