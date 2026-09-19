from collections.abc import Callable as Callable
from typing import Any, Generic, ParamSpec, TypeVar

import dataclasses

from _typeshed import Incomplete
from jax._src import util as util

safe_map: Incomplete
A = ParamSpec("A")
K = TypeVar("K")

@dataclasses.dataclass
class Fusion(Generic[A, K]):
    func: Callable[A, K]
    in_type: tuple[tuple[Any, ...], dict[str, Any]]
    out_type: Any
    def __call__(self, *args: A.args, **kwargs: A.kwargs) -> K: ...
    @property
    def shape(self): ...
    @property
    def dtype(self): ...
    @property
    def type(self): ...
    @property
    def in_shape(self): ...
    @property
    def in_dtype(self): ...
