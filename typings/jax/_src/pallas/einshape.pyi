from typing import Literal, NamedTuple

import dataclasses

from _typeshed import Incomplete
from jax._src import (
    api as api,
    core as jax_core,
    dispatch as dispatch,
    hijax as hijax,
    typing as jax_typing,
)
from jax._src.frozen_dict import FrozenDict as FrozenDict
from jax._src.interpreters import mlir as mlir
from jax._src.lax import lax as lax
from jax._src.pallas.mosaic import tpu_info as tpu_info

@dataclasses.dataclass(frozen=True)
class SplitDims:
    index: int
    sizes: tuple[int, ...]
    def transform_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]: ...

@dataclasses.dataclass(frozen=True)
class MergeDims:
    index: int
    count: int
    def transform_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]: ...

@dataclasses.dataclass(frozen=True)
class Transpose:
    permutation: tuple[int, ...]
    def transform_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]: ...

type Transform = SplitDims | MergeDims | Transpose

def get_einshape_transforms(
    equation: str,
    input_shape: tuple[int, ...],
    **sizes: int,
) -> list[Transform]: ...

einshape_lo_p: Incomplete

def einshape_lo(
    equation: str,
    x: jax_typing.Array,
    assert_is_tile_preserving: bool,
    **sizes: int,
) -> jax_typing.Array: ...

class Einshape(hijax.VJPHiPrimitive):
    in_avals: Incomplete
    out_aval: Incomplete
    equation: Incomplete
    sizes: Incomplete
    assert_is_tile_preserving: Incomplete
    params: Incomplete
    def __init__(
        self,
        x_aval: jax_core.ShapedArray,
        *,
        equation: str,
        assert_is_tile_preserving: bool,
        sizes: dict[str, int],
    ) -> None: ...
    def expand(self, x: jax_typing.Array) -> jax_typing.Array: ...

def einshape(
    equation: str,
    x: jax_typing.Array,
    assert_is_tile_preserving: bool = False,
    **sizes: int,
) -> jax_typing.Array: ...

class Factor(NamedTuple):
    size: int
    kind: Literal["outer", "sublane", "lane"]
