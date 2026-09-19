import dataclasses

from jax._src import (
    core as core,
    tree_util as tree_util,
)
from jax._src.typing import Array as Array

@dataclasses.dataclass
class Slice:
    start: int | Array
    size: int | Array
    stride: int = ...
    def __post_init__(self) -> None: ...
    @property
    def is_dynamic_start(self): ...
    @property
    def is_dynamic_size(self): ...
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Slice: ...
    @classmethod
    def from_slice(cls, slc: slice, size: int) -> Slice: ...

def dslice(
    start: int | Array | None,
    size: int | Array | None = None,
    stride: int | None = None,
) -> slice | Slice: ...

ds = dslice
