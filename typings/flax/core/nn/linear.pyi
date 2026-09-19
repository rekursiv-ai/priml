from _typeshed import Incomplete
from flax import struct as struct
from flax.core import Scope as Scope
from flax.linen import initializers as initializers

import numpy as np

default_kernel_init: Incomplete

def dense_general(
    scope,
    inputs,
    features,
    axis: int = -1,
    batch_dims=(),
    bias: bool = True,
    dtype=...,
    kernel_init=...,
    bias_init=...,
    precision=None,
): ...
def dense(
    scope,
    inputs,
    features,
    bias: bool = True,
    dtype=...,
    precision=None,
    kernel_init=...,
    bias_init=...,
): ...
def conv(
    scope,
    inputs,
    features,
    kernel_size,
    strides=None,
    padding: str = "SAME",
    input_dilation=None,
    kernel_dilation=None,
    feature_group_count: int = 1,
    bias: bool = True,
    dtype=...,
    precision=None,
    kernel_init=...,
    bias_init=...,
): ...
def conv_transpose(
    scope,
    inputs,
    features,
    kernel_size,
    strides=None,
    padding: str = "SAME",
    kernel_dilation=None,
    bias: bool = True,
    dtype=...,
    precision=None,
    kernel_init=...,
    bias_init=...,
): ...

default_embed_init: Incomplete

@struct.dataclass
class Embedding:
    table: np.ndarray
    def lookup(self, indices): ...
    def attend(self, query): ...

def embedding(
    scope: Scope,
    num_embeddings: int,
    features: int,
    init_fn=...,
) -> Embedding: ...
