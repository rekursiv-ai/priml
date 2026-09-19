from collections.abc import Hashable, Sequence

from _typeshed import Incomplete
from jax._src import (
    config as config,
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
    prng as prng,
    xla_bridge as xla_bridge,
)
from jax._src.api import (
    jit as jit,
    vmap as vmap,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)
from jax._src.lax import lax as lax
from jax._src.mesh import get_abstract_mesh as get_abstract_mesh
from jax._src.numpy.util import (
    check_arraylike as check_arraylike,
    promote_dtypes_inexact as promote_dtypes_inexact,
)
from jax._src.pjit import auto_axes as auto_axes
from jax._src.sharding_impls import (
    NamedSharding as NamedSharding,
    canonicalize_sharding as canonicalize_sharding,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DType as DType,
    DTypeLike as DTypeLike,
)
from jax._src.util import canonicalize_axis as canonicalize_axis

RealArray = ArrayLike
IntegerArray = ArrayLike
DTypeLikeInt = DTypeLike
DTypeLikeUInt = DTypeLike
DTypeLikeFloat = DTypeLike
type Shape = Sequence[int]
PRNGImpl: Incomplete
UINT_DTYPES: Incomplete

def default_prng_impl(): ...

class PRNGSpec:
    def __init__(self, impl) -> None: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other) -> bool: ...

type PRNGSpecDesc = str | PRNGSpec | PRNGImpl | Hashable

def resolve_prng_impl(impl_spec: PRNGSpecDesc | None) -> PRNGImpl: ...
def key(seed: int | ArrayLike, *, impl: PRNGSpecDesc | None = None) -> Array: ...
def PRNGKey(seed: int | ArrayLike, *, impl: PRNGSpecDesc | None = None) -> Array: ...
def fold_in(key: ArrayLike, data: IntegerArray) -> Array: ...
def split(key: ArrayLike, num: int | tuple[int, ...] = 2) -> Array: ...
def key_impl(keys: ArrayLike) -> str | PRNGSpec: ...
def key_data(keys: ArrayLike) -> Array: ...
def wrap_key_data(key_bits_array: Array, *, impl: PRNGSpecDesc | None = None): ...
def maybe_auto_axes(f, out_sharding, **hoist_kwargs): ...
def bits(
    key: ArrayLike,
    shape: Shape = (),
    dtype: DTypeLikeUInt | None = None,
    *,
    out_sharding=None,
) -> Array: ...
def canonicalize_sharding_for_samplers(out_sharding, name, shape): ...
def uniform(
    key: ArrayLike,
    shape: Shape = (),
    dtype: DTypeLikeFloat | None = None,
    minval: RealArray = 0.0,
    maxval: RealArray = 1.0,
    *,
    out_sharding=None,
) -> Array: ...
def randint(
    key: ArrayLike,
    shape: Shape,
    minval: IntegerArray,
    maxval: IntegerArray,
    dtype: DTypeLikeInt | None = None,
    *,
    out_sharding=None,
) -> Array: ...
def permutation(
    key: ArrayLike,
    x: int | ArrayLike,
    axis: int = 0,
    independent: bool = False,
    *,
    out_sharding=None,
) -> Array: ...
def choice(
    key: ArrayLike,
    a: int | ArrayLike,
    shape: Shape = (),
    replace: bool = True,
    p: RealArray | None = None,
    axis: int = 0,
    mode: str | None = None,
) -> Array: ...
def normal(
    key: ArrayLike,
    shape: Shape = (),
    dtype: DTypeLikeFloat | None = None,
    *,
    out_sharding=None,
) -> Array: ...
def multivariate_normal(
    key: ArrayLike,
    mean: RealArray,
    cov: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
    method: str = "cholesky",
) -> Array: ...
def truncated_normal(
    key: ArrayLike,
    lower: RealArray,
    upper: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
    *,
    out_sharding=None,
) -> Array: ...
def bernoulli(
    key: ArrayLike,
    p: RealArray = 0.5,
    shape: Shape | None = None,
    mode: str = "low",
    *,
    out_sharding=None,
) -> Array: ...
def beta(
    key: ArrayLike,
    a: RealArray,
    b: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def cauchy(
    key: ArrayLike,
    shape: Shape = (),
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def dirichlet(
    key: ArrayLike,
    alpha: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def exponential(
    key: ArrayLike,
    shape: Shape = (),
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...

random_gamma_p: Incomplete

def gamma(
    key: ArrayLike,
    a: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def loggamma(
    key: ArrayLike,
    a: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def poisson(
    key: ArrayLike,
    lam: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeInt | None = None,
) -> Array: ...
def gumbel(
    key: ArrayLike,
    shape: Shape = (),
    dtype: DTypeLikeFloat | None = None,
    mode: str | None = None,
    *,
    out_sharding=None,
) -> Array: ...
def categorical(
    key: ArrayLike,
    logits: RealArray,
    axis: int = -1,
    shape: Shape | None = None,
    replace: bool = True,
    mode: str | None = None,
) -> Array: ...
def laplace(
    key: ArrayLike,
    shape: Shape = (),
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def logistic(
    key: ArrayLike,
    shape: Shape = (),
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def pareto(
    key: ArrayLike,
    b: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def t(
    key: ArrayLike,
    df: RealArray,
    shape: Shape = (),
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def chisquare(
    key: ArrayLike,
    df: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def f(
    key: ArrayLike,
    dfnum: RealArray,
    dfden: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def rademacher(
    key: ArrayLike,
    shape: Shape = (),
    dtype: DTypeLikeInt | None = None,
) -> Array: ...
def maxwell(
    key: ArrayLike,
    shape: Shape = (),
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def double_sided_maxwell(
    key: ArrayLike,
    loc: RealArray,
    scale: RealArray,
    shape: Shape = (),
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def weibull_min(
    key: ArrayLike,
    scale: RealArray,
    concentration: RealArray,
    shape: Shape = (),
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def orthogonal(
    key: ArrayLike,
    n: int,
    shape: Shape = (),
    dtype: DTypeLikeFloat | None = None,
    m: int | None = None,
) -> Array: ...
def generalized_normal(
    key: ArrayLike,
    p: float,
    shape: Shape = (),
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def ball(
    key: ArrayLike,
    d: int,
    p: float = 2,
    shape: Shape = (),
    dtype: DTypeLikeFloat | None = None,
): ...
def rayleigh(
    key: ArrayLike,
    scale: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def wald(
    key: ArrayLike,
    mean: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def geometric(
    key: ArrayLike,
    p: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeInt | None = None,
) -> Array: ...
def triangular(
    key: ArrayLike,
    left: RealArray,
    mode: RealArray,
    right: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def lognormal(
    key: ArrayLike,
    sigma: RealArray = ...,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...
def binomial(
    key: Array,
    n: RealArray,
    p: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
) -> Array: ...

random_clone_p: Incomplete

def multinomial(
    key: Array,
    n: RealArray,
    p: RealArray,
    *,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
    unroll: int | bool = 1,
): ...
def clone(key): ...
def random_insert_pvary(name, key, *args): ...
