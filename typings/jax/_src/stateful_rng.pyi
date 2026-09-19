from collections.abc import Sequence

import dataclasses

from jax._src import (
    api_util as api_util,
    core as core,
    dtypes as dtypes,
    random as random,
    ref as ref,
    tree_util as tree_util,
    typing as typing,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DTypeLike as DTypeLike,
)

@dataclasses.dataclass(frozen=True)
class StatefulPRNG:
    def __post_init__(self) -> None: ...
    def key(self, shape: int | Sequence[int] = ()) -> Array: ...
    def random(
        self,
        size: int | Sequence[int] | None = None,
        dtype: DTypeLike = ...,
    ): ...
    def uniform(
        self,
        low: ArrayLike = 0,
        high: ArrayLike = 1,
        size: int | Sequence[int] | None = None,
        *,
        dtype: DTypeLike = ...,
    ) -> Array: ...
    def normal(
        self,
        loc: ArrayLike = 0,
        scale: ArrayLike = 1,
        size: int | Sequence[int] | None = None,
        *,
        dtype: DTypeLike = ...,
    ) -> Array: ...
    def integers(
        self,
        low: ArrayLike,
        high: ArrayLike | None = None,
        size: int | Sequence[int] | None = None,
        *,
        dtype: DTypeLike = ...,
    ) -> Array: ...
    def split(self, num: int | Sequence[int]) -> StatefulPRNG: ...
    def spawn(self, n_children: int) -> list[StatefulPRNG]: ...

def stateful_rng(
    seed: typing.ArrayLike | None = None,
    *,
    impl: random.PRNGSpecDesc | None = None,
) -> StatefulPRNG: ...
