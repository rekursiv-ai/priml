from collections.abc import (
    Callable as Callable,
    Generator,
    Sequence,
)
from typing import Any, NamedTuple, TypeVar, overload

import dataclasses
import enum

from _typeshed import Incomplete
from jax._src import (
    ad_util as ad_util,
    api as api,
    api_util as api_util,
    array as array,
    config as config,
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
    effects as effects,
    literals as literals,
    pjit as pjit,
    source_info_util as source_info_util,
    tree_util as tree_util,
    util as util,
)
from jax._src.abstract_arrays import array_types as array_types
from jax._src.core import (
    Primitive as Primitive,
    ShapedArray as ShapedArray,
    abstract_token as abstract_token,
    canonicalize_shape as canonicalize_shape,
    typeof as typeof,
)
from jax._src.errors import UnexpectedTracerError as UnexpectedTracerError
from jax._src.hashable_array import HashableArray as HashableArray
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
    pxla as pxla,
    remat as remat,
)
from jax._src.lax import slicing as slicing
from jax._src.lax.utils import (
    dtype_to_string as dtype_to_string,
    input_dtype as input_dtype,
    standard_multi_result_abstract_eval as standard_multi_result_abstract_eval,
    standard_primitive as standard_primitive,
)
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import (
    chlo as chlo,
    hlo as hlo,
)
from jax._src.mesh import (
    get_abstract_mesh as get_abstract_mesh,
    get_concrete_mesh as get_concrete_mesh,
)
from jax._src.sharding import Sharding as Sharding
from jax._src.sharding_impls import (
    NamedSharding as NamedSharding,
    PartitionSpec as P,
    PmapSharding as PmapSharding,
    canonicalize_sharding as canonicalize_sharding,
    flatten_spec as flatten_spec,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DimSize as DimSize,
    DType as DType,
    DTypeLike as DTypeLike,
    DuckTypedArray as DuckTypedArray,
    Shape as Shape,
)
from jax._src.util import (
    cache as cache,
    canonicalize_axis as canonicalize_axis,
    foreach as foreach,
    safe_map as safe_map,
    safe_zip as safe_zip,
    split_list as split_list,
    tuple_insert as tuple_insert,
    weakref_lru_cache as weakref_lru_cache,
)

import numpy as np

T = TypeVar("T")
map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
export: Incomplete

def asarray(x: ArrayLike) -> Array: ...
@overload
def broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]: ...
@overload
def broadcast_shapes(
    *shapes: tuple[int | core.Tracer, ...],
) -> tuple[int | core.Tracer, ...]: ...
def broadcast_shardings(*avals): ...
@export
def neg(x: ArrayLike) -> Array: ...
@export
def sign(x: ArrayLike) -> Array: ...
@export
def nextafter(x1: ArrayLike, x2: ArrayLike) -> Array: ...
@export
def floor(x: ArrayLike) -> Array: ...
@export
def ceil(x: ArrayLike) -> Array: ...

class RoundingMethod(enum.IntEnum):
    AWAY_FROM_ZERO = 0
    TO_NEAREST_EVEN = 1

@export
def round(x: ArrayLike, rounding_method: RoundingMethod = ...) -> Array: ...
@export
def is_finite(x: ArrayLike) -> Array: ...

class Tolerance:
    atol: Incomplete
    rtol: Incomplete
    ulps: Incomplete
    def __init__(self, atol: float = 0.0, rtol: float = 0.0, ulps: int = 0) -> None: ...

class AccuracyMode(enum.Enum):
    HIGHEST = 1
    DEFAULT = 2

@export
def exp(x: ArrayLike, accuracy=None) -> Array: ...
def exp2(x: ArrayLike, accuracy=None) -> Array: ...
@export
def expm1(x: ArrayLike, accuracy=None) -> Array: ...
@export
def log(x: ArrayLike, accuracy=None) -> Array: ...
@export
def log1p(x: ArrayLike, accuracy=None) -> Array: ...
@export
def tanh(x: ArrayLike, accuracy=None) -> Array: ...
@export
def logistic(x: ArrayLike, accuracy=None) -> Array: ...
@export
def sin(x: ArrayLike, accuracy=None) -> Array: ...
@export
def cos(x: ArrayLike, accuracy=None) -> Array: ...
@export
def atan2(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def real(x: ArrayLike) -> Array: ...
@export
def imag(x: ArrayLike) -> Array: ...
@export
def complex(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def conj(x: ArrayLike) -> Array: ...
@export
def abs(x: ArrayLike) -> Array: ...
@export
def pow(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def integer_pow(x: ArrayLike, y: int) -> Array: ...
@export
def sqrt(x: ArrayLike, accuracy=None) -> Array: ...
@export
def rsqrt(x: ArrayLike, accuracy=None) -> Array: ...
@export
def cbrt(x: ArrayLike, accuracy=None) -> Array: ...
@export
def bitwise_not(x: ArrayLike) -> Array: ...
@export
def bitwise_and(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def bitwise_or(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def bitwise_xor(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def population_count(x: ArrayLike) -> Array: ...
@export
def clz(x: ArrayLike) -> Array: ...
@export
def add(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def sub(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def mul(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def div(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def rem(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def max(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def min(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def shift_left(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def shift_right_arithmetic(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def shift_right_logical(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def eq(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def ne(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def ge(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def gt(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def le(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def lt(x: ArrayLike, y: ArrayLike) -> Array: ...
@export
def convert_element_type(
    operand: ArrayLike,
    new_dtype: DTypeLike | dtypes.ExtendedDType,
) -> Array: ...
@export
def bitcast_convert_type(operand: ArrayLike, new_dtype: DTypeLike) -> Array: ...
def clamp(min: ArrayLike, x: ArrayLike, max: ArrayLike) -> Array: ...
def composite(decomposition: Callable, name: str, version: int = 0): ...
def composite_jvp(*args, **_) -> None: ...
def composite_transpose(*args, **_) -> None: ...

composite_p: Incomplete

def concatenate(operands: Array | Sequence[ArrayLike], dimension: int) -> Array: ...
def split(
    operand: ArrayLike,
    sizes: Sequence[DimSize],
    axis: int = 0,
) -> Sequence[Array]: ...

class Precision(enum.Enum):
    DEFAULT = 0
    HIGH = 1
    HIGHEST = 2

class DotAlgorithm(NamedTuple):
    lhs_precision_type: DTypeLike
    rhs_precision_type: DTypeLike
    accumulation_type: DTypeLike
    lhs_component_count: int = ...
    rhs_component_count: int = ...
    num_primitive_operations: int = ...
    allow_imprecise_accumulation: bool = ...

class DotAlgorithmPreset(enum.Enum):
    DEFAULT = ...
    ANY_F8_ANY_F8_F32 = ...
    ANY_F8_ANY_F8_F32_FAST_ACCUM = ...
    ANY_F8_ANY_F8_ANY = ...
    ANY_F8_ANY_F8_ANY_FAST_ACCUM = ...
    F16_F16_F16 = ...
    F16_F16_F32 = ...
    BF16_BF16_BF16 = ...
    BF16_BF16_F32 = ...
    BF16_BF16_F32_X3 = ...
    BF16_BF16_F32_X6 = ...
    BF16_BF16_F32_X9 = ...
    TF32_TF32_F32 = ...
    TF32_TF32_F32_X3 = ...
    F32_F32_F32 = ...
    F64_F64_F64 = ...
    @property
    def supported_lhs_types(self) -> tuple[DTypeLike, ...] | None: ...
    @property
    def supported_rhs_types(self) -> tuple[DTypeLike, ...] | None: ...
    @property
    def accumulation_type(self) -> DTypeLike | None: ...
    def supported_output_types(
        self,
        lhs_dtype: DTypeLike,
        rhs_dtype: DTypeLike,
    ) -> tuple[DTypeLike, ...] | None: ...

type PrecisionLike = (
    str
    | Precision
    | tuple[str, str]
    | tuple[Precision, Precision]
    | DotAlgorithm
    | DotAlgorithmPreset
    | None
)
type CanonicalPrecision = (
    tuple[Precision, Precision] | DotAlgorithm | DotAlgorithmPreset | None
)
type DotDimensionNumbers = tuple[
    tuple[Sequence[int], Sequence[int]],
    tuple[Sequence[int], Sequence[int]],
]

def dot_general(
    lhs: ArrayLike,
    rhs: ArrayLike,
    dimension_numbers: DotDimensionNumbers,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    *,
    out_sharding=None,
) -> Array: ...
def dot(
    lhs: ArrayLike,
    rhs: ArrayLike,
    *args,
    dimension_numbers: DotDimensionNumbers | None = None,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    out_sharding=None,
) -> Array: ...
def ragged_dot(
    lhs: Array,
    rhs: Array,
    group_sizes: Array,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    group_offset: Array | None = None,
) -> Array: ...

@dataclasses.dataclass(frozen=True)
class RaggedDotDimensionNumbers:
    dot_dimension_numbers: DotDimensionNumbers
    lhs_ragged_dimensions: Sequence[int]
    rhs_group_dimensions: Sequence[int]
    def __init__(
        self,
        dot_dimension_numbers,
        lhs_ragged_dimensions,
        rhs_group_dimensions,
    ) -> None: ...

def ragged_dot_general(
    lhs: Array,
    rhs: Array,
    group_sizes: Array,
    ragged_dot_dimension_numbers: RaggedDotDimensionNumbers,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    group_offset: Array | None = None,
) -> Array: ...
def broadcast(
    operand: ArrayLike,
    sizes: Sequence[int],
    *,
    out_sharding=None,
) -> Array: ...
def broadcast_in_dim(
    operand: ArrayLike,
    shape: Shape,
    broadcast_dimensions: Sequence[int],
    *,
    out_sharding=None,
) -> Array: ...
def broadcast_to_rank(x: ArrayLike, rank: int) -> Array: ...
def tile(operand: ArrayLike, reps: Sequence[int]) -> Array: ...
def reshape(
    operand: ArrayLike,
    new_sizes: Shape,
    dimensions: Sequence[int] | None = None,
    *,
    out_sharding: NamedSharding | P | None = None,
) -> Array: ...
def pad(
    operand: ArrayLike,
    padding_value: ArrayLike,
    padding_config: Sequence[tuple[int, int, int]],
) -> Array: ...
def rev(operand: ArrayLike, dimensions: Sequence[int]) -> Array: ...
def select(pred: ArrayLike, on_true: ArrayLike, on_false: ArrayLike) -> Array: ...
def select_n(which: ArrayLike, *cases: ArrayLike) -> Array: ...
def transpose(operand: ArrayLike, permutation: Sequence[int] | np.ndarray) -> Array: ...
def argmin(operand: ArrayLike, axis: int, index_dtype: DTypeLike) -> Array: ...
def argmax(operand: ArrayLike, axis: int, index_dtype: DTypeLike) -> Array: ...
def reduce(
    operands: Any,
    init_values: Any,
    computation: Callable[[Any, Any], Any],
    dimensions: Sequence[int],
    out_sharding: NamedSharding | P | None = None,
) -> Any: ...
def reduce_sum(
    operand: ArrayLike,
    axes: Sequence[int],
    *,
    out_sharding=None,
) -> Array: ...
def reduce_prod(operand: ArrayLike, axes: Sequence[int]) -> Array: ...
def reduce_max(operand: ArrayLike, axes: Sequence[int]) -> Array: ...
def reduce_min(operand: ArrayLike, axes: Sequence[int]) -> Array: ...
def reduce_or(operand: ArrayLike, axes: Sequence[int]) -> Array: ...
def reduce_and(operand: ArrayLike, axes: Sequence[int]) -> Array: ...
def reduce_xor(operand: ArrayLike, axes: Sequence[int]) -> Array: ...
@overload
def sort(
    operand: Array,
    dimension: int = -1,
    is_stable: bool = True,
    num_keys: int = 1,
) -> Array: ...
@overload
def sort(
    operand: Sequence[Array],
    dimension: int = -1,
    is_stable: bool = True,
    num_keys: int = 1,
) -> tuple[Array, ...]: ...
def sort_key_val(
    keys: Array,
    values: ArrayLike,
    dimension: int = -1,
    is_stable: bool = True,
) -> tuple[Array, Array]: ...
def top_k(operand: ArrayLike, k: int, *, axis: int = -1) -> tuple[Array, Array]: ...
def full(
    shape: Shape,
    fill_value: ArrayLike,
    dtype: DTypeLike | None = None,
    *,
    sharding: Sharding | None = None,
) -> Array: ...
def zeros_like_shaped_array(aval: ShapedArray) -> Array: ...
def iota(dtype: DTypeLike, size: int) -> Array: ...
def broadcasted_iota(
    dtype: DTypeLike,
    shape: Shape,
    dimension: int,
    *,
    out_sharding=None,
) -> Array: ...
def stop_gradient(x: T) -> T: ...
def reduce_precision(
    operand: float | ArrayLike,
    exponent_bits: int,
    mantissa_bits: int,
) -> Array: ...
def squeeze(array: ArrayLike, dimensions: Sequence[int]) -> Array: ...
def expand_dims(array: ArrayLike, dimensions: Sequence[int]) -> Array: ...
def full_like(
    x: ArrayLike | DuckTypedArray,
    fill_value: ArrayLike,
    dtype: DTypeLike | None = None,
    shape: Shape | None = None,
    sharding: Sharding | None = None,
) -> Array: ...
def collapse(
    operand: Array,
    start_dimension: int,
    stop_dimension: int | None = None,
) -> Array: ...
def batch_matmul(lhs: Array, rhs: Array, precision: PrecisionLike = None) -> Array: ...
def square(x: ArrayLike) -> Array: ...
def reciprocal(x: ArrayLike) -> Array: ...
@export
def tan(x: ArrayLike, accuracy=None) -> Array: ...
@export
def asin(x: ArrayLike) -> Array: ...
@export
def acos(x: ArrayLike) -> Array: ...
@export
def atan(x: ArrayLike) -> Array: ...
@export
def sinh(x: ArrayLike) -> Array: ...
@export
def cosh(x: ArrayLike) -> Array: ...
@export
def asinh(x: ArrayLike) -> Array: ...
@export
def acosh(x: ArrayLike) -> Array: ...
@export
def atanh(x: ArrayLike) -> Array: ...
def unop_dtype_rule(
    result_dtype,
    accepted_dtypes,
    name,
    aval,
    supports_narrow_ints: bool = True,
    **kwargs,
): ...
def unop_reduced_rule(out_s, aval, **kwargs): ...
def unop(result_dtype, accepted_dtypes, name, supports_narrow_ints: bool = True): ...

standard_unop: Incomplete

def naryop_dtype_rule(
    result_dtype,
    accepted_dtypes,
    name,
    *avals,
    require_same: bool = True,
    allow_extended_dtype: bool = False,
    **kwargs,
): ...
def broadcasting_shape_rule(name, *avals): ...
def broadcasting_sharding_rule(name, *avals): ...
def nary_reduced_rule(out_s, *avals, **params): ...
def naryop(
    result_dtype,
    accepted_dtypes,
    name,
    allow_extended_dtype: bool = False,
    require_same_dtypes: bool = True,
    unreduced_rule=None,
    reduced_rule=None,
): ...

standard_naryop: Incomplete
neg_p: Incomplete
sign_p: Incomplete
nextafter_p: Incomplete
floor_p: Incomplete
ceil_p: Incomplete
round_p: Incomplete
is_finite_p: Incomplete
exp_p: Incomplete
exp2_p: Incomplete
log_p: Incomplete
expm1_p: Incomplete
log1p_p: Incomplete
tanh_p: Incomplete
logistic_p: Incomplete

def logistic_impl(x, accuracy): ...

sin_p: Incomplete
cos_p: Incomplete
tan_p: Incomplete
asin_p: Incomplete
acos_p: Incomplete
atan_p: Incomplete
atan2_p: Incomplete
sinh_p: Incomplete
cosh_p: Incomplete
asinh_p: Incomplete
acosh_p: Incomplete
atanh_p: Incomplete
real_p: Incomplete
imag_p: Incomplete
complex_p: Incomplete
conj_p: Incomplete
abs_p: Incomplete
sqrt_p: Incomplete
rsqrt_p: Incomplete
cbrt_p: Incomplete
square_p: Incomplete
pow_p: Incomplete
integer_pow_p: Incomplete
not_p: Incomplete
and_p: Incomplete
or_p: Incomplete
xor_p: Incomplete
population_count_p: Incomplete
clz_p: Incomplete
add_p: Primitive
sub_p: Incomplete
mul_p: Incomplete
div_p: Incomplete
rem_p: Incomplete
max_p: core.Primitive
min_p: core.Primitive
shift_left_p: Incomplete
shift_right_arithmetic_p: Incomplete
shift_right_logical_p: Incomplete
eq_p: Incomplete
ne_p: Incomplete
ge_p: Incomplete
gt_p: Incomplete
le_p: Incomplete
lt_p: Incomplete
eq_to_p: Incomplete
le_to_p: Incomplete
lt_to_p: Incomplete
convert_element_type_p: Incomplete
to_edtype_p: Incomplete
from_edtype_p: Incomplete
bitcast_convert_type_p: Incomplete

def tuple_delete(tup, idx): ...

dot_general_p: Incomplete

def precision_attr(precision: Precision) -> ir.ArrayAttr: ...
def chlo_precision_attr(precision: Precision) -> ir.ArrayAttr: ...
def dot_algorithm_attr(
    precision: CanonicalPrecision,
    lhs_dtype: DTypeLike,
    rhs_dtype: DTypeLike,
) -> hlo.DotAlgorithm | None: ...
def get_algorithm_compute_types(
    algorithm: DotAlgorithm | DotAlgorithmPreset,
    lhs_dtype: DTypeLike,
    rhs_dtype: DTypeLike,
    out_dtype: DTypeLike,
) -> tuple[DTypeLike, DTypeLike, DTypeLike]: ...
def accuracy_attr(accuracy) -> hlo.ResultAccuracyAttr: ...

class RaggedDotMode(enum.Enum):
    RAGGED_NONCONTRACTING = 1
    RAGGED_CONTRACTING = 2
    RAGGED_BATCH = 3

ragged_dot_general_p: Incomplete
broadcast_in_dim_p: Incomplete
tile_p: Incomplete
clamp_p: Incomplete
concatenate_p: Incomplete
split_p: Incomplete
pad_p: Incomplete
squeeze_p: Incomplete

def shape_as_value(shape: core.Shape): ...

reshape_p: Incomplete
rev_p: Incomplete
transpose_p: Incomplete
select_n_p: Incomplete
reduce_p: Incomplete
reduce_sum_p: Incomplete
reduce_prod_p: Incomplete
reduce_max_p: Incomplete
reduce_min_p: Incomplete

class _ArgMinMaxReducer:
    def __init__(self, value_comparator: Callable[[Any, Any], Any]) -> None: ...
    def __call__(self, op_val_index, acc_val_index): ...

argmin_p: Incomplete
argmax_p: Incomplete
reduce_or_p: Incomplete
reduce_and_p: Incomplete
reduce_xor_p: Incomplete
reduce_precision_p: Incomplete
sort_p: Incomplete
top_k_p: Incomplete

def create_token(_=None): ...

create_token_p: Incomplete

def after_all(*operands): ...

after_all_p: Incomplete

def rng_uniform(a, b, shape): ...

rng_uniform_p: Incomplete

class RandomAlgorithm(enum.IntEnum):
    RNG_DEFAULT = 0
    RNG_THREE_FRY = 1
    RNG_PHILOX = 2

rng_bit_generator_p: Incomplete
copy_p: Incomplete

def dce_sink(val) -> None: ...

class NoDCEEffect(effects.Effect):
    def __hash__(self): ...
    def __eq__(self, other): ...

no_dce_effect: Incomplete
dce_sink_p: Incomplete

def rng_bit_generator(key, shape, dtype=..., algorithm=..., *, out_sharding=None): ...

iota_p: Incomplete

class PaddingType(enum.Enum):
    VALID = 1
    SAME = 2
    SAME_LOWER = 3

def padtype_to_pads(
    in_shape: Sequence[int] | np.ndarray,
    window_shape: Sequence[int] | np.ndarray,
    window_strides: Sequence[int] | np.ndarray,
    padding: str | PaddingType,
) -> list[tuple[int, int]]: ...
def check_same_dtypes(name: str, *avals: ShapedArray) -> None: ...

dtype: Callable

def ranges_like(*xs) -> Generator[Incomplete]: ...
def remaining(original, *removed_lists): ...
def canonicalize_precision(precision: PrecisionLike) -> CanonicalPrecision: ...
def empty(shape, dtype, *, out_sharding=None): ...

empty_p: Incomplete

def empty2(dtype, *, memory_space): ...

empty2_p: Incomplete
tie_p: Incomplete

def optimization_barrier(operand, /): ...

optimization_barrier_p: Incomplete
