from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import TypeAlias, overload

import enum

from _typeshed import Incomplete
from jax import (
    api_util as api_util,
    lax as lax,
)
from jax._src import (
    core as jax_core,
    dtypes as dtypes,
    effects as effects,
)
from jax._src.api_util import (
    check_no_transformed_refs_args as check_no_transformed_refs_args,
)
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import (
    arith as arith,
    scf as scf,
    vector as vector,
)
from jax._src.pallas.mosaic import (
    sc_core as sc_core,
    sc_lowering as sc_lowering,
)
from jax._src.state import (
    indexing as indexing,
    types as state_types,
)
from jax.experimental.mosaic.dialects import tpu as tpu

import jax

TransformedRef: TypeAlias
Ref: TypeAlias
load_p: Incomplete

def load_expanded(ref: Ref, *, mask: jax.Array) -> jax.Array: ...

swap_p: Incomplete

def store_compressed(ref: Ref, x: jax.Array, *, mask: jax.Array) -> None: ...
def addupdate(ref: Ref, x: jax.Array) -> None: ...
def addupdate_compressed(ref: Ref, x: jax.Array, *, mask: jax.Array) -> None: ...

gather_p: Incomplete

def load_gather(
    ref: Ref,
    indices: Sequence[jax.Array],
    *,
    mask: jax.Array | None = None,
) -> jax.Array: ...

scatter_p: Incomplete

def store_scatter(
    ref: Ref,
    indices: Sequence[jax.Array],
    x: jax.Array,
    *,
    mask: jax.Array | None = None,
) -> None: ...
def addupdate_scatter(
    ref: Ref,
    indices: Sequence[jax.Array],
    x: jax.Array,
    *,
    mask: jax.Array | None = None,
) -> None: ...

bitcast_p: Incomplete

def bitcast(x: jax.Array, dtype: jax.typing.DTypeLike) -> jax.Array: ...

class MemoryEffect(jax_core.Effect): ...

barrier_p: Incomplete

def subcore_barrier() -> None: ...

scan_count_p: Incomplete

def scan_count(
    x: jax.Array,
    mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]: ...

masked_cummax_p: Incomplete
masked_cummin_p: Incomplete
masked_cumsum_p: Incomplete

def cummax(x: jax.Array, *, mask: jax.Array | None = None) -> jax.Array: ...
def cummin(x: jax.Array, *, mask: jax.Array | None = None) -> jax.Array: ...
def cumsum(x: jax.Array, *, mask: jax.Array | None = None) -> jax.Array: ...

masked_sort_p: Incomplete

def sort_key_val(
    keys: jax.Array,
    values: jax.Array,
    *,
    mask: jax.Array | None = None,
    descending: bool = False,
) -> jax.Array: ...

parallel_loop_p: Incomplete

@overload
def parallel_loop(
    lower: jax.typing.ArrayLike,
    upper: jax.typing.ArrayLike,
    step: jax.typing.ArrayLike = ...,
    *,
    unroll: int = ...,
    carry: None = None,
) -> Callable[[Callable[[jax.Array], None]], None]: ...
@overload
def parallel_loop(
    lower: jax.typing.ArrayLike,
    upper: jax.typing.ArrayLike,
    step: jax.typing.ArrayLike = ...,
    *,
    unroll: int = ...,
    carry: _T,
) -> Callable[[Callable[[jax.Array, _T], _T]], _T]: ...

class PackFormat(enum.Enum):
    COMPRESSED = "compressed"
    INTERLEAVED = "interleaved"

pack_p: Incomplete

def pack(
    a: jax.Array,
    b: jax.Array,
    /,
    *,
    format: PackFormat,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array: ...

unpack_p: Incomplete

def unpack(
    ab: jax.Array,
    /,
    *,
    format: PackFormat,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> tuple[jax.Array, jax.Array]: ...

all_reduce_population_count_p: Incomplete

def all_reduce_population_count(x: jax.Array, *, reduce: int = 1) -> jax.Array: ...

all_reduce_ffs_p: Incomplete

def all_reduce_ffs(x: jax.Array, *, reduce: int = 1) -> jax.Array: ...

fetch_and_add_p: Incomplete

def fetch_and_add(
    x_ref: jax.Ref | state_types.TransformedRef,
    value: jax.typing.ArrayLike,
    *,
    subcore_id: jax.typing.ArrayLike,
) -> jax.Array: ...
