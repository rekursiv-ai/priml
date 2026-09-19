import typing as tp

from _typeshed import Incomplete
from flax import nnx as nnx
from flax.core.frozen_dict import FrozenDict as FrozenDict
from flax.nnx import (
    rnglib as rnglib,
    variablelib as variablelib,
)
from flax.nnx.module import (
    Module as Module,
    first_from as first_from,
)
from flax.nnx.nn import (
    dtypes as dtypes,
    initializers as initializers,
)
from flax.typing import (
    ConvGeneralDilatedT as ConvGeneralDilatedT,
    DotGeneralT as DotGeneralT,
    Dtype as Dtype,
    EinsumT as EinsumT,
    Initializer as Initializer,
    LaxPadding as LaxPadding,
    PaddingLike as PaddingLike,
    PrecisionLike as PrecisionLike,
    PromoteDtypeFn as PromoteDtypeFn,
    Shape as Shape,
)

import jax

Array: Incomplete
Axis = int
Size = int
default_kernel_init: Incomplete
default_bias_init: Incomplete

def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding: ...

class LinearGeneral(Module):
    in_features: Incomplete
    out_features: Incomplete
    axis: Incomplete
    batch_axis: Incomplete
    use_bias: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    precision: Incomplete
    dot_general: Incomplete
    dot_general_cls: Incomplete
    promote_dtype: Incomplete
    preferred_element_type: Incomplete
    kernel: Incomplete
    bias: nnx.Param[jax.Array] | None
    def __init__(
        self,
        in_features: Size | tp.Sequence[Size],
        out_features: Size | tp.Sequence[Size],
        *,
        axis: Axis | tp.Sequence[Axis] = -1,
        batch_axis: tp.Mapping[Axis, Size] = ...,
        use_bias: bool = True,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        kernel_init: Initializer = ...,
        bias_init: Initializer = ...,
        precision: PrecisionLike = None,
        promote_dtype: PromoteDtypeFn = ...,
        dot_general: DotGeneralT | None = None,
        dot_general_cls: tp.Any = None,
        preferred_element_type: Dtype | None = None,
        rngs: rnglib.Rngs,
        kernel_metadata: tp.Mapping[str, tp.Any] = ...,
        bias_metadata: tp.Mapping[str, tp.Any] = ...,
    ) -> None: ...
    def __call__(self, inputs: Array, out_sharding=None) -> Array: ...

class Linear(Module):
    kernel: Incomplete
    bias: nnx.Param[jax.Array] | None
    in_features: Incomplete
    out_features: Incomplete
    use_bias: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    precision: Incomplete
    dot_general: Incomplete
    promote_dtype: Incomplete
    preferred_element_type: Incomplete
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        precision: PrecisionLike = None,
        kernel_init: Initializer = ...,
        bias_init: Initializer = ...,
        dot_general: DotGeneralT = ...,
        promote_dtype: PromoteDtypeFn = ...,
        preferred_element_type: Dtype | None = None,
        rngs: rnglib.Rngs,
        kernel_metadata: tp.Mapping[str, tp.Any] = ...,
        bias_metadata: tp.Mapping[str, tp.Any] = ...,
    ) -> None: ...
    def __call__(self, inputs: Array, out_sharding=None) -> Array: ...

class Einsum(Module):
    kernel: Incomplete
    bias: nnx.Param | None
    einsum_str: Incomplete
    kernel_shape: Incomplete
    bias_shape: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    precision: Incomplete
    promote_dtype: Incomplete
    einsum_op: Incomplete
    preferred_element_type: Incomplete
    def __init__(
        self,
        einsum_str: str,
        kernel_shape: Shape,
        bias_shape: Shape | None = None,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        precision: PrecisionLike = None,
        kernel_init: Initializer = ...,
        bias_init: Initializer = ...,
        promote_dtype: PromoteDtypeFn = ...,
        einsum_op: EinsumT = ...,
        preferred_element_type: Dtype | None = None,
        rngs: rnglib.Rngs,
        kernel_metadata: tp.Mapping[str, tp.Any] = ...,
        bias_metadata: tp.Mapping[str, tp.Any] = ...,
    ) -> None: ...
    def __call__(
        self,
        inputs: Array,
        einsum_str: str | None = None,
        out_sharding=None,
    ) -> Array: ...

class Conv(Module):
    kernel_shape: Incomplete
    kernel: Incomplete
    bias: nnx.Param[jax.Array] | None
    in_features: Incomplete
    out_features: Incomplete
    kernel_size: Incomplete
    strides: Incomplete
    padding: Incomplete
    input_dilation: Incomplete
    kernel_dilation: Incomplete
    feature_group_count: Incomplete
    use_bias: Incomplete
    mask: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    precision: Incomplete
    conv_general_dilated: Incomplete
    promote_dtype: Incomplete
    preferred_element_type: Incomplete
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tp.Sequence[int],
        strides: int | tp.Sequence[int] | None = 1,
        *,
        padding: PaddingLike = "SAME",
        input_dilation: int | tp.Sequence[int] | None = 1,
        kernel_dilation: int | tp.Sequence[int] | None = 1,
        feature_group_count: int = 1,
        use_bias: bool = True,
        mask: Array | None = None,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        precision: PrecisionLike = None,
        kernel_init: Initializer = ...,
        bias_init: Initializer = ...,
        conv_general_dilated: ConvGeneralDilatedT = ...,
        promote_dtype: PromoteDtypeFn = ...,
        preferred_element_type: Dtype | None = None,
        rngs: rnglib.Rngs,
        kernel_metadata: tp.Mapping[str, tp.Any] = ...,
        bias_metadata: tp.Mapping[str, tp.Any] = ...,
    ) -> None: ...
    def __call__(self, inputs: Array, out_sharding=None) -> Array: ...

class ConvTranspose(Module):
    kernel_size: Incomplete
    in_features: Incomplete
    out_features: Incomplete
    strides: Incomplete
    padding: Incomplete
    kernel_dilation: Incomplete
    use_bias: Incomplete
    mask: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    precision: Incomplete
    transpose_kernel: Incomplete
    promote_dtype: Incomplete
    preferred_element_type: Incomplete
    kernel_shape: Incomplete
    kernel: Incomplete
    bias: nnx.Param | None
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tp.Sequence[int],
        strides: int | tp.Sequence[int] | None = None,
        *,
        padding: PaddingLike = "SAME",
        kernel_dilation: int | tp.Sequence[int] | None = None,
        use_bias: bool = True,
        mask: Array | None = None,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        precision: PrecisionLike | None = None,
        kernel_init: Initializer = ...,
        bias_init: Initializer = ...,
        transpose_kernel: bool = False,
        promote_dtype: PromoteDtypeFn = ...,
        preferred_element_type: Dtype | None = None,
        rngs: rnglib.Rngs,
        kernel_metadata: tp.Mapping[str, tp.Any] = ...,
        bias_metadata: tp.Mapping[str, tp.Any] = ...,
    ) -> None: ...
    def __call__(self, inputs: Array) -> Array: ...

default_embed_init: Incomplete

class Embed(Module):
    embedding: Incomplete
    num_embeddings: Incomplete
    features: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    promote_dtype: Incomplete
    def __init__(
        self,
        num_embeddings: int,
        features: int,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        embedding_init: Initializer = ...,
        promote_dtype: PromoteDtypeFn = ...,
        rngs: rnglib.Rngs,
        embedding_metadata: tp.Mapping[str, tp.Any] = ...,
    ) -> None: ...
    def __call__(self, inputs: Array, out_sharding=None) -> Array: ...
    def attend(self, query: Array, out_sharding=None) -> Array: ...
