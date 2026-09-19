import typing as tp

from _typeshed import Incomplete
from flax import nnx as nnx
from flax.nnx import rnglib as rnglib
from flax.nnx.module import (
    Module as Module,
    first_from as first_from,
)
from flax.nnx.nn import (
    dtypes as dtypes,
    initializers as initializers,
)
from flax.typing import (
    Array as Array,
    Axes as Axes,
    Dtype as Dtype,
    Initializer as Initializer,
    PromoteDtypeFn as PromoteDtypeFn,
)

import jax

class BatchNorm(Module):
    mean: Incomplete
    var: Incomplete
    scale: nnx.Param[jax.Array] | None
    bias: nnx.Param[jax.Array] | None
    num_features: Incomplete
    use_running_average: Incomplete
    axis: Incomplete
    momentum: Incomplete
    epsilon: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    use_bias: Incomplete
    use_scale: Incomplete
    axis_name: Incomplete
    axis_index_groups: Incomplete
    use_fast_variance: Incomplete
    promote_dtype: Incomplete
    def __init__(
        self,
        num_features: int,
        *,
        use_running_average: bool = False,
        axis: int = -1,
        momentum: float = 0.99,
        epsilon: float = 1e-05,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: Initializer = ...,
        scale_init: Initializer = ...,
        axis_name: str | None = None,
        axis_index_groups: tp.Any = None,
        use_fast_variance: bool = True,
        promote_dtype: PromoteDtypeFn = ...,
        rngs: rnglib.Rngs,
        bias_metadata: tp.Mapping[str, tp.Any] = ...,
        scale_metadata: tp.Mapping[str, tp.Any] = ...,
    ) -> None: ...
    def __call__(
        self,
        x,
        use_running_average: bool | None = None,
        *,
        mask: jax.Array | None = None,
    ): ...
    def set_view(self, use_running_average: bool | None = None): ...

class LayerNorm(Module):
    scale: nnx.Param[jax.Array] | None
    bias: nnx.Param[jax.Array] | None
    num_features: Incomplete
    epsilon: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    use_bias: Incomplete
    use_scale: Incomplete
    reduction_axes: Incomplete
    feature_axes: Incomplete
    axis_name: Incomplete
    axis_index_groups: Incomplete
    use_fast_variance: Incomplete
    promote_dtype: Incomplete
    def __init__(
        self,
        num_features: int,
        *,
        epsilon: float = 1e-06,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: Initializer = ...,
        scale_init: Initializer = ...,
        reduction_axes: Axes = -1,
        feature_axes: Axes = -1,
        axis_name: str | None = None,
        axis_index_groups: tp.Any = None,
        use_fast_variance: bool = True,
        promote_dtype: PromoteDtypeFn = ...,
        rngs: rnglib.Rngs,
        bias_metadata: tp.Mapping[str, tp.Any] = ...,
        scale_metadata: tp.Mapping[str, tp.Any] = ...,
    ) -> None: ...
    def __call__(self, x, *, mask: jax.Array | None = None): ...

class RMSNorm(Module):
    scale: nnx.Param[jax.Array] | None
    num_features: Incomplete
    epsilon: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    use_scale: Incomplete
    reduction_axes: Incomplete
    feature_axes: Incomplete
    axis_name: Incomplete
    axis_index_groups: Incomplete
    use_fast_variance: Incomplete
    promote_dtype: Incomplete
    def __init__(
        self,
        num_features: int,
        *,
        epsilon: float = 1e-06,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        use_scale: bool = True,
        scale_init: Initializer = ...,
        reduction_axes: Axes = -1,
        feature_axes: Axes = -1,
        axis_name: str | None = None,
        axis_index_groups: tp.Any = None,
        use_fast_variance: bool = True,
        promote_dtype: PromoteDtypeFn = ...,
        rngs: rnglib.Rngs,
        scale_metadata: tp.Mapping[str, tp.Any] = ...,
    ) -> None: ...
    def __call__(self, x, mask: jax.Array | None = None): ...

class GroupNorm(Module):
    feature_axis: int
    num_groups: Incomplete
    group_size: Incomplete
    scale: nnx.Param[jax.Array] | None
    bias: nnx.Param[jax.Array] | None
    epsilon: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    use_bias: Incomplete
    use_scale: Incomplete
    reduction_axes: Incomplete
    axis_name: Incomplete
    axis_index_groups: Incomplete
    use_fast_variance: Incomplete
    promote_dtype: Incomplete
    def __init__(
        self,
        num_features: int,
        num_groups: int | None = 32,
        group_size: int | None = None,
        *,
        epsilon: float = 1e-06,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: Initializer = ...,
        scale_init: Initializer = ...,
        reduction_axes: Axes | None = None,
        axis_name: str | None = None,
        axis_index_groups: tp.Any = None,
        use_fast_variance: bool = True,
        promote_dtype: PromoteDtypeFn = ...,
        rngs: rnglib.Rngs,
        bias_metadata: tp.Mapping[str, tp.Any] = ...,
        scale_metadata: tp.Mapping[str, tp.Any] = ...,
    ) -> None: ...
    def __call__(self, x, *, mask: jax.Array | None = None): ...

class WeightNorm(nnx.Module):
    layer_instance: Incomplete
    feature_axes: Incomplete
    use_scale: Incomplete
    scale_init: Incomplete
    epsilon: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    variable_filter: Incomplete
    promote_dtype: Incomplete
    scales: dict | None
    def __init__(
        self,
        layer_instance: nnx.Module,
        *,
        feature_axes: Axes | None = -1,
        use_scale: bool = True,
        scale_init: Initializer = ...,
        epsilon: float = 1e-12,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        variable_filter: nnx.filterlib.Filter = ...,
        promote_dtype: PromoteDtypeFn = ...,
        rngs: rnglib.Rngs,
    ) -> None: ...
    def __call__(self, x: Array, *args, **kwargs) -> Array: ...

class InstanceNorm(Module):
    scale: nnx.Param[jax.Array] | None
    bias: nnx.Param[jax.Array] | None
    num_features: Incomplete
    epsilon: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    use_bias: Incomplete
    use_scale: Incomplete
    feature_axes: Incomplete
    axis_name: Incomplete
    axis_index_groups: Incomplete
    use_fast_variance: Incomplete
    promote_dtype: Incomplete
    def __init__(
        self,
        num_features: int,
        *,
        epsilon: float = 1e-06,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: Initializer = ...,
        scale_init: Initializer = ...,
        feature_axes: Axes = -1,
        axis_name: str | None = None,
        axis_index_groups: tp.Any = None,
        use_fast_variance: bool = True,
        promote_dtype: PromoteDtypeFn = ...,
        rngs: rnglib.Rngs,
        bias_metadata: tp.Mapping[str, tp.Any] = ...,
        scale_metadata: tp.Mapping[str, tp.Any] = ...,
    ) -> None: ...
    def __call__(self, x, *, mask: jax.Array | None = None): ...

class SpectralNorm(Module):
    layer_instance: Incomplete
    n_steps: Incomplete
    epsilon: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    error_on_non_matrix: Incomplete
    use_running_average: Incomplete
    batch_stats: Incomplete
    def __init__(
        self,
        layer_instance: Module,
        *,
        n_steps: int = 1,
        epsilon: float = 1e-12,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        error_on_non_matrix: bool = False,
        update_stats: bool = True,
        rngs: rnglib.Rngs,
    ) -> None: ...
    def __call__(self, x, update_stats: bool | None = None): ...
