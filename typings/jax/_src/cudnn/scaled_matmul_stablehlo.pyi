from dataclasses import dataclass

from jax._src import (
    api as api,
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
    tree_util as tree_util,
)
from jax._src.custom_derivatives import custom_vjp as custom_vjp
from jax._src.custom_partitioning import custom_partitioning as custom_partitioning
from jax._src.interpreters import (
    batching as batching,
    mlir as mlir,
)
from jax._src.interpreters.mlir import ir as ir
from jax._src.lax import lax as lax
from jax._src.lax.lax import (
    ranges_like as ranges_like,
    remaining as remaining,
)
from jax._src.sharding_impls import NamedSharding as NamedSharding
from jax._src.typing import (
    Array as Array,
    DTypeLike as DTypeLike,
)

block_scaled_dot_name: str

@dataclass
class BlockScaleConfig:
    mode: str
    block_size: int
    data_type: DTypeLike
    scale_type: DTypeLike
    global_scale: Array | None
    infer_only: bool

def default_layouts(*shapes): ...
def element_type_to_backend_config_type(dtype): ...
def scaled_matmul_wrapper(
    lhs: Array,
    rhs: Array,
    lhs_scales: Array,
    rhs_scales: Array,
    preferred_element_type: DTypeLike = ...,
) -> Array: ...
def shape_normalization(x, dimension_numbers): ...
def compute_dot_output_shape(
    lhs_shape,
    rhs_shape,
    lhs_dimension_numbers,
    rhs_dimension_numbers,
): ...
def cast_to_e8m0_with_rounding_up(x): ...
def e8m0_to_dtype(x, dtype): ...
def quantize(x, config): ...
def scaled_dot_impl(lhs, rhs, dimension_numbers, preferred_element_type, configs): ...
def scaled_dot_general_transpose_lhs(
    g,
    x,
    y,
    *,
    dimension_numbers,
    preferred_element_type,
    configs,
    swap_ans: bool = False,
): ...
def scaled_dot_general_transpose_rhs(
    g,
    x,
    y,
    *,
    dimension_numbers,
    preferred_element_type: DTypeLike,
    configs: list[BlockScaleConfig],
): ...
def scaled_dot_general_fn(
    lhs,
    rhs,
    dimension_numbers,
    preferred_element_type,
    configs,
): ...
def scaled_dot_fwd(lhs, rhs, dimension_numbers, preferred_element_type, configs): ...
def scaled_dot_bwd(dimension_numbers, preferred_element_type, configs, res, g): ...
def ensure_tuple(dimension_numbers): ...
def scaled_dot_general_wrapper(
    lhs,
    rhs,
    dimension_numbers,
    preferred_element_type=...,
    configs: list[BlockScaleConfig] | None = None,
): ...
