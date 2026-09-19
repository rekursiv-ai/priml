from collections.abc import Sequence
from typing import Any, Literal, Protocol

from _typeshed import Incomplete
from jax._src import (
    core as core,
    dtypes as dtypes,
    random as random,
)
from jax._src.named_sharding import NamedSharding as NamedSharding
from jax._src.partition_spec import PartitionSpec as PartitionSpec
from jax._src.sharding_impls import canonicalize_sharding as canonicalize_sharding
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DType as DType,
)
from jax._src.util import set_module as set_module

export: Incomplete
type DTypeLikeFloat = Any
type DTypeLikeComplex = Any
type DTypeLikeInexact = Any
type RealNumeric = Any
type OutShardingType = NamedSharding | PartitionSpec | None

class Initializer(Protocol):
    def __call__(
        self,
        key: Array,
        shape: core.Shape,
        dtype: DTypeLikeInexact | None = None,
        out_sharding: OutShardingType = None,
    ) -> Array: ...

@export
def zeros(
    key: Array,
    shape: core.Shape,
    dtype: DTypeLikeInexact | None = None,
    out_sharding: OutShardingType = None,
) -> Array: ...
@export
def ones(
    key: Array,
    shape: core.Shape,
    dtype: DTypeLikeInexact | None = None,
    out_sharding: OutShardingType = None,
) -> Array: ...
@export
def constant(
    value: ArrayLike,
    dtype: DTypeLikeInexact | None = None,
) -> Initializer: ...
@export
def uniform(
    scale: RealNumeric = 0.01,
    dtype: DTypeLikeInexact | None = None,
) -> Initializer: ...
@export
def normal(
    stddev: RealNumeric = 0.01,
    dtype: DTypeLikeInexact | None = None,
) -> Initializer: ...
@export
def truncated_normal(
    stddev: RealNumeric = 0.01,
    dtype: DTypeLikeInexact | None = None,
    lower: RealNumeric = -2.0,
    upper: RealNumeric = 2.0,
) -> Initializer: ...
@export
def variance_scaling(
    scale: RealNumeric,
    mode: Literal["fan_in", "fan_out", "fan_avg", "fan_geo_avg"],
    distribution: Literal["truncated_normal", "normal", "uniform"],
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DTypeLikeInexact | None = None,
) -> Initializer: ...
@export
def glorot_uniform(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DTypeLikeInexact | None = None,
) -> Initializer: ...

xavier_uniform = glorot_uniform

@export
def glorot_normal(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DTypeLikeInexact | None = None,
) -> Initializer: ...

xavier_normal = glorot_normal

@export
def lecun_uniform(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DTypeLikeInexact | None = None,
) -> Initializer: ...
@export
def lecun_normal(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DTypeLikeInexact | None = None,
) -> Initializer: ...
@export
def he_uniform(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DTypeLikeInexact | None = None,
) -> Initializer: ...

kaiming_uniform = he_uniform

@export
def he_normal(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DTypeLikeInexact | None = None,
) -> Initializer: ...

kaiming_normal = he_normal

@export
def orthogonal(
    scale: RealNumeric = 1.0,
    column_axis: int = -1,
    dtype: DTypeLikeInexact | None = None,
) -> Initializer: ...
@export
def delta_orthogonal(
    scale: RealNumeric = 1.0,
    column_axis: int = -1,
    dtype: DTypeLikeInexact | None = None,
) -> Initializer: ...
