from typing import NamedTuple

from _typeshed import Incomplete
from jax._src import (
    dtypes as dtypes,
    util as util,
)

neg: Incomplete
sign: Incomplete
floor: Incomplete
ceil: Incomplete

def round(x): ...

nextafter: Incomplete
is_finite: Incomplete
exp: Incomplete
exp2: Incomplete
expm1: Incomplete
log: Incomplete
log1p: Incomplete
tanh: Incomplete
sin: Incomplete
cos: Incomplete
atan2: Incomplete
sqrt: Incomplete
rsqrt: Incomplete
cbrt: Incomplete
square: Incomplete
reciprocal: Incomplete
tan: Incomplete
asin: Incomplete
acos: Incomplete
atan: Incomplete
sinh: Incomplete
cosh: Incomplete
asinh: Incomplete
acosh: Incomplete
atanh: Incomplete

def logistic(x): ...
def betainc(a, b, x): ...
def lgamma(x): ...
def digamma(x): ...

igamma: Incomplete
igammac: Incomplete

def erf(x): ...
def erfc(x): ...
def erf_inv(x): ...
def bessel_i0e(x): ...
def bessel_i1e(x): ...

real: Incomplete
imag: Incomplete

def conj(x): ...
def complex(x, y): ...

abs: Incomplete
pow: Incomplete
bitwise_not: Incomplete
bitwise_and: Incomplete
bitwise_or: Incomplete
bitwise_xor: Incomplete
add: Incomplete
sub: Incomplete
mul: Incomplete

def div(lhs, rhs): ...
def rem(lhs, rhs): ...

max: Incomplete
min: Incomplete
shift_left: Incomplete
shift_right_arithmetic: Incomplete

def population_count(x): ...
def clz(x): ...

eq: Incomplete
ne: Incomplete
ge: Incomplete
gt: Incomplete
le: Incomplete
lt: Incomplete

def convert_element_type(operand, dtype): ...
def bitcast_convert_type(operand, dtype): ...
def clamp(min, operand, max): ...
def concatenate(operands, dimension): ...
def conv(lhs, rhs, window_strides, padding): ...
def conv_with_general_padding(
    lhs,
    rhs,
    window_strides,
    padding,
    lhs_dilation,
    rhs_dilation,
): ...
def conv_general_dilated(
    lhs,
    rhs,
    window_strides,
    padding,
    lhs_dilation,
    rhs_dilation,
    dimension_numbers,
): ...

dot: Incomplete

def dot_general(lhs, rhs, dimension_numbers): ...
def ragged_dot(lhs, rhs, group_sizes): ...
def broadcast(operand, sizes): ...
def broadcast_in_dim(operand, shape, broadcast_dimensions): ...

sum: Incomplete
squeeze: Incomplete

def reshape(operand, new_sizes, dimensions=None): ...
def pad(operand, padding_value, padding_config): ...
def rev(operand, dimensions): ...

select: Incomplete

def slice(operand, start_indices, limit_indices, strides=None): ...
def dynamic_slice(operand, start_indices, slice_sizes): ...
def dynamic_update_slice(operand, update, start_indices): ...

transpose: Incomplete

def reduce(operand, init_value, computation, dimensions): ...
def reduce_window(
    operand,
    init_value,
    computation,
    window_dimensions,
    window_strides,
    padding,
    base_dilation,
): ...

sort: Incomplete

def sort_key_val(keys, values, dimension: int = -1): ...
def padtype_to_pads(in_shape, filter_shape, window_strides, padding): ...

class MonoidRecord(NamedTuple):
    reducer: Incomplete
    identity: Incomplete

def top_k(operand, k, axis: int = -1): ...
