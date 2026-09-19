from dataclasses import dataclass

from _typeshed import Incomplete
from jax._src import (
    config as config,
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
    tree_util as tree_util,
)
from jax._src.core import (
    AxisName as AxisName,
    ShapedArray as ShapedArray,
    abstract_token as abstract_token,
    check_unreduced_args as check_unreduced_args,
    pvary as pvary,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
    pxla as pxla,
)
from jax._src.lax import (
    control_flow as control_flow,
    lax as lax,
    slicing as slicing,
)
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import hlo as hlo
from jax._src.mesh import get_abstract_mesh as get_abstract_mesh
from jax._src.sharding_impls import (
    NamedSharding as NamedSharding,
    ShardingContext as ShardingContext,
    SPMDAxisContext as SPMDAxisContext,
)
from jax._src.typing import Array as Array
from jax._src.util import (
    canonicalize_axis as canonicalize_axis,
    moveaxis as moveaxis,
    safe_map as safe_map,
    safe_zip as safe_zip,
    unzip2 as unzip2,
)

unsafe_map: Incomplete
map: Incomplete
unsafe_zip: Incomplete
zip: Incomplete

def psum(x, axis_name, *, axis_index_groups=None): ...
def pmean(x, axis_name, *, axis_index_groups=None): ...
def pmax(x, axis_name, *, axis_index_groups=None): ...
def pmin(x, axis_name, *, axis_index_groups=None): ...
def pargmin(x, axis_name): ...
def pargmax(x, axis_name): ...
def pbroadcast(x, axis_name, source): ...
def ppermute(x, axis_name, perm): ...
def psend(x, axis_name, perm): ...
def precv(token, out_shape, axis_name, perm): ...
def pshuffle(x, axis_name, perm): ...
def pswapaxes(x, axis_name, axis, *, axis_index_groups=None): ...
def all_to_all(
    x,
    axis_name,
    split_axis,
    concat_axis,
    *,
    axis_index_groups=None,
    tiled: bool = False,
): ...
def ragged_all_to_all(
    operand,
    output,
    input_offsets,
    send_sizes,
    output_offsets,
    recv_sizes,
    *,
    axis_name,
    axis_index_groups=None,
): ...
def axis_index(axis_name: AxisName) -> Array: ...
def axis_size(axis_name: AxisName) -> int: ...

psum_p: Incomplete
pmax_p: Incomplete
pmin_p: Incomplete
ppermute_p: Incomplete

@dataclass(frozen=True)
class SingleSideCollectiveEffect(core.Effect):
    def __hash__(self): ...
    def __eq__(self, other): ...

single_side_collective_effect: Incomplete
psend_p: Incomplete
precv_p: Incomplete
pbroadcast_p: Incomplete
all_to_all_p: Incomplete
ragged_all_to_all_p: Incomplete

def insert_collective_pvary(axis_name, x): ...
def all_gather(
    x,
    axis_name,
    *,
    axis_index_groups=None,
    axis: int = 0,
    tiled: bool = False,
    to: str = "varying",
): ...
def collective_vma_rule(prim_name, axis_name, x_aval): ...

all_gather_p: Incomplete

def all_gather_invariant(x, axis_name, *, axis: int = 0, tiled: bool = False): ...

all_gather_invariant_p: Incomplete
reduce_scatter_p: Incomplete

def psum_scatter(
    x,
    axis_name,
    *,
    scatter_dimension: int = 0,
    axis_index_groups=None,
    tiled: bool = False,
): ...

axis_index_p: Incomplete

def bind_psum_invariant(leaf, *, axes, axis_index_groups): ...

psum_invariant_p: Incomplete

def all_gather_reduced(x, axis_name, *, axis: int = 0, tiled: bool = False): ...

all_gather_reduced_p: Incomplete

def unreduced_psum_scatter(
    x,
    axis_name,
    *,
    scatter_dimension: int = 0,
    tiled: bool = False,
): ...

unreduced_reduce_scatter_p: Incomplete

def unreduced_psum(x, axis_name): ...

unreduced_psum_p: Incomplete

def preduced(x, axis_name): ...

preduced_p: Incomplete

def vary_unreduced_cast(x, axis_name): ...

vary_unreduced_cast_p: Incomplete

def pcast(x, axis_name, *, to: str): ...
def all_gather_start(x, axis_name, axis: int = 0, tiled: bool = False): ...

all_gather_start_p: Incomplete

def all_gather_done(x): ...

all_gather_done_p: Incomplete
