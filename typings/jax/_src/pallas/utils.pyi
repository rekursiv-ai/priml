from typing import Any, overload

import dataclasses

from jax._src import (
    core as jax_core,
    dtypes as dtypes,
    typing as jax_typing,
)
from jax._src.lax import lax as lax
from jax._src.util import split_list as split_list

@overload
def cdiv(a: int, b: int) -> int: ...
@overload
def cdiv(a: int, b: jax_typing.Array) -> jax_typing.Array: ...
@overload
def cdiv(a: jax_typing.Array, b: int) -> jax_typing.Array: ...
@overload
def cdiv(a: jax_typing.Array, b: jax_typing.Array) -> jax_typing.Array: ...
def strides_from_shape(shape: tuple[int, ...]) -> tuple[int, ...]: ...
def next_power_of_2(x: int) -> int: ...
def pattern_match_scan_to_fori_loop(
    jaxpr: jax_core.Jaxpr,
    num_consts: int,
    num_carry: int,
) -> tuple[jax_core.Jaxpr, bool]: ...
def pattern_match_while_to_fori_loop(
    cond_jaxpr: jax_core.ClosedJaxpr,
    cond_nconsts: int,
    body_jaxpr: jax_core.ClosedJaxpr,
    body_nconsts: int,
) -> tuple[jax_core.Jaxpr | None, str | None]: ...
def erf_inv_lowering_helper(x): ...
def sign_lowering_helper(x): ...
def nextafter_lowering_helper(x, y): ...

@dataclasses.dataclass(frozen=True)
class MeshInfo:
    mesh_shape: tuple[int, ...]
    axis_names: tuple[str, ...]
    mesh_strides: tuple[int, ...]
    @staticmethod
    def from_mesh(mesh: Any) -> MeshInfo: ...
