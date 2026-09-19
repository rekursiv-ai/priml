from collections.abc import Generator

import typing as tp

from _typeshed import Incomplete
from flax import (
    struct as struct,
    typing as typing,
)
from flax.nnx import (
    filterlib as filterlib,
    graphlib as graphlib,
)
from flax.nnx.nn import initializers as initializers
from flax.nnx.pytreelib import Pytree as Pytree
from flax.nnx.variablelib import Variable as Variable
from flax.typing import (
    MISSING as MISSING,
    Key as Key,
    Missing as Missing,
)

import jax

F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
A = tp.TypeVar("A")
type Counts = list[int]
type AxesValue = int | None
type SplitPattern = AxesValue | tuple[AxesValue, ...]
OutShardingType: tp.TypeAlias
Fargs = tp.ParamSpec("Fargs")

class KeylessInitializer(tp.Protocol):
    def __call__(
        self,
        shape: typing.Shape,
        dtype: tp.Any | None = None,
        out_sharding: OutShardingType = None,
    ) -> jax.Array: ...

class RngState(Variable[jax.Array]):
    tag: str

class RngCount(RngState): ...
class RngKey(RngState): ...

NotKey: Incomplete

class RngStream(Pytree):
    tag: Incomplete
    key: Incomplete
    count: Incomplete
    def __init__(self, key: jax.Array | int, *, tag: str) -> None: ...
    def __call__(self) -> jax.Array: ...
    def fork(self, *, split: int | tuple[int, ...] | None = None): ...
    bits: Incomplete
    uniform: Incomplete
    randint: Incomplete
    permutation: Incomplete
    choice: Incomplete
    normal: Incomplete
    multivariate_normal: Incomplete
    truncated_normal: Incomplete
    bernoulli: Incomplete
    beta: Incomplete
    cauchy: Incomplete
    dirichlet: Incomplete
    exponential: Incomplete
    gamma: Incomplete
    loggamma: Incomplete
    poisson: Incomplete
    gumbel: Incomplete
    categorical: Incomplete
    laplace: Incomplete
    logistic: Incomplete
    pareto: Incomplete
    t: Incomplete
    chisquare: Incomplete
    f: Incomplete
    rademacher: Incomplete
    maxwell: Incomplete
    double_sided_maxwell: Incomplete
    weibull_min: Incomplete
    orthogonal: Incomplete
    generalized_normal: Incomplete
    ball: Incomplete
    rayleigh: Incomplete
    wald: Incomplete
    geometric: Incomplete
    triangular: Incomplete
    lognormal: Incomplete
    binomial: Incomplete
    multinomial: Incomplete
    delta_orthogonal: Incomplete
    glorot_normal: Incomplete
    glorot_uniform: Incomplete
    he_normal: Incomplete
    he_uniform: Incomplete
    kaiming_normal: Incomplete
    kaiming_uniform: Incomplete
    lecun_normal: Incomplete
    lecun_uniform: Incomplete
    variance_scaling: Incomplete
    xavier_normal: Incomplete
    xavier_uniform: Incomplete

RngValue: Incomplete

class Rngs(Pytree):
    def __init__(
        self,
        default: RngValue
        | RngStream
        | tp.Mapping[str, RngValue | RngStream]
        | None = None,
        **rngs: RngValue | RngStream,
    ) -> None: ...
    def __getitem__(self, name: str): ...
    def __getattr__(self, name: str): ...
    def __call__(self): ...
    def __iter__(self) -> tp.Iterator[str]: ...
    def __len__(self) -> int: ...
    def __contains__(self, name: tp.Any) -> bool: ...
    def items(self) -> Generator[Incomplete]: ...
    def fork(
        self,
        /,
        *,
        split: tp.Mapping[filterlib.Filter, int | tuple[int, ...]]
        | int
        | tuple[int, ...]
        | None = None,
    ): ...
    bits: Incomplete
    uniform: Incomplete
    randint: Incomplete
    permutation: Incomplete
    choice: Incomplete
    normal: Incomplete
    multivariate_normal: Incomplete
    truncated_normal: Incomplete
    bernoulli: Incomplete
    beta: Incomplete
    cauchy: Incomplete
    dirichlet: Incomplete
    exponential: Incomplete
    gamma: Incomplete
    loggamma: Incomplete
    poisson: Incomplete
    gumbel: Incomplete
    categorical: Incomplete
    laplace: Incomplete
    logistic: Incomplete
    pareto: Incomplete
    t: Incomplete
    chisquare: Incomplete
    f: Incomplete
    rademacher: Incomplete
    maxwell: Incomplete
    double_sided_maxwell: Incomplete
    weibull_min: Incomplete
    orthogonal: Incomplete
    generalized_normal: Incomplete
    ball: Incomplete
    rayleigh: Incomplete
    wald: Incomplete
    geometric: Incomplete
    triangular: Incomplete
    lognormal: Incomplete
    binomial: Incomplete
    multinomial: Incomplete
    delta_orthogonal: Incomplete
    glorot_normal: Incomplete
    glorot_uniform: Incomplete
    he_normal: Incomplete
    he_uniform: Incomplete
    kaiming_normal: Incomplete
    kaiming_uniform: Incomplete
    lecun_normal: Incomplete
    lecun_uniform: Incomplete
    variance_scaling: Incomplete
    xavier_normal: Incomplete
    xavier_uniform: Incomplete

StreamBackup: Incomplete

class SplitBackups(struct.PyTreeNode, tp.Iterable[StreamBackup]):
    backups: list[StreamBackup]
    def __iter__(self) -> tp.Iterator[StreamBackup]: ...
    def __enter__(self): ...
    def __exit__(self, *args) -> None: ...

@tp.overload
def split_rngs(
    node: tp.Any,
    /,
    *,
    splits: int | tuple[int, ...],
    only: filterlib.Filter = ...,
    squeeze: bool = False,
    graph: tp.Literal[True] | None = None,
) -> SplitBackups: ...
@tp.overload
def split_rngs(
    node: A,
    /,
    *,
    splits: int | tuple[int, ...],
    only: filterlib.Filter = ...,
    squeeze: bool = False,
    graph: tp.Literal[False],
) -> A: ...
@tp.overload
def split_rngs(
    *,
    splits: int | tuple[int, ...],
    only: filterlib.Filter = ...,
    squeeze: bool = False,
    graph: bool | None = None,
) -> tp.Callable[[F], F]: ...
@tp.overload
def fork_rngs(
    node: tp.Any,
    /,
    *,
    split: tp.Mapping[filterlib.Filter, int | tuple[int, ...] | None]
    | int
    | None = None,
    graph: bool | None = None,
) -> SplitBackups: ...
@tp.overload
def fork_rngs(
    *,
    split: tp.Mapping[filterlib.Filter, int | tuple[int, ...] | None]
    | int
    | None = None,
    graph: bool | None = None,
) -> tp.Callable[[F], F]: ...
def backup_keys(node: tp.Any, /, *, graph: bool | None = None): ...
def reseed(
    node,
    /,
    *,
    graph: bool | None = None,
    policy: tp.Literal["scalars_only", "match_shape"]
    | tp.Callable[[tuple, jax.Array, tuple[int, ...]], jax.Array] = "scalars_only",
    **stream_keys: RngValue,
): ...
def restore_rngs(backups: tp.Iterable[StreamBackup], /): ...
