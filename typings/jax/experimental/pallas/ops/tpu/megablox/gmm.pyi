from collections.abc import Callable
from typing import Any

import functools

from jax import lax as lax
from jax.experimental.pallas.ops.tpu.megablox import common as common

import jax.numpy as jnp

partial = functools.partial
type GroupMetadata = Any

def make_group_metadata(
    *,
    group_sizes: jnp.ndarray,
    m: int,
    tm: int,
    start_group: jnp.ndarray,
    num_nonzero_groups: int,
    visit_empty_groups: bool = True,
) -> GroupMetadata: ...

type LutFn = Callable[[int, int, int], tuple[int, int, int] | None]

def gmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = ...,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> jnp.ndarray: ...
def tgmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = ...,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    num_actual_groups: int | None = None,
    existing_out: jnp.ndarray | None = None,
    interpret: bool = False,
) -> jnp.ndarray: ...
