from typing import Any

import types

from _typeshed import Incomplete
from jax._src import (
    ad_util as ad_util,
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
    source_info_util as source_info_util,
    traceback_util as traceback_util,
    tree_util as tree_util,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)
from jax._src.lax import lax as lax
from jax._src.state import indexing as indexing
from jax._src.state.types import (
    AbstractLinVal as AbstractLinVal,
    AbstractRef as AbstractRef,
    AccumEffect as AccumEffect,
    ReadEffect as ReadEffect,
    Transform as Transform,
    TransformedRef as TransformedRef,
    WriteEffect as WriteEffect,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)
from jax._src.util import (
    safe_map as safe_map,
    safe_zip as safe_zip,
)

type HijaxType = Any
map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
get_p: Incomplete
type Indexer = int | slice | Array | types.EllipsisType

def get_ref_and_transforms(
    ref_or_view: Any,
    idx: Indexer | tuple[Indexer, ...] | None,
    function_name: str,
) -> tuple[Any, tuple[Transform, ...]]: ...
def ref_get(
    ref: core.Ref | TransformedRef,
    idx: Indexer | tuple[Indexer, ...] | None = None,
) -> Array | HijaxType: ...

swap_p: Incomplete

def ref_swap(
    ref: core.Ref | TransformedRef,
    idx: Indexer | tuple[Indexer, ...] | None,
    value: ArrayLike | HijaxType,
    _function_name: str = "ref_swap",
) -> Array | HijaxType: ...
def ref_set(
    ref: core.Ref | TransformedRef,
    idx: Indexer | tuple[Indexer, ...] | None,
    value: ArrayLike | HijaxType,
) -> None: ...

addupdate_p: Incomplete

def ref_addupdate(
    ref: core.Ref | TransformedRef,
    idx: Indexer | tuple[Indexer, ...] | None,
    x: ArrayLike | HijaxType,
) -> None: ...

pp_ref_var: Incomplete

def pp_ref_transforms(context: core.JaxprPpContext, ref, transforms): ...
def addupdate_jvp_rule(primals: list[Any], tangents: list[Any], **params: Any): ...
def addupdate_transpose_fancy(cts_in, ref_, x, *idx, **params) -> None: ...

broadcast_to_p: Incomplete

def broadcast_to(a: Array, shape: tuple[int, ...]) -> Array: ...
def create_linear(ty, memory_space=None): ...

create_linear_p: Incomplete

def pin(x): ...

pin_p: Incomplete

def unpin(x): ...

unpin_p: Incomplete
