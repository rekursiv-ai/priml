from collections.abc import Callable as Callable

from flax import struct as struct
from flax.core import Scope as Scope
from flax.linen import initializers as initializers

import numpy as np

from .linear import (
    default_kernel_init as default_kernel_init,
    dense_general as dense_general,
)

def dot_product_attention(
    scope,
    query,
    key,
    value,
    dtype=...,
    bias=None,
    axis=None,
    broadcast_dropout: bool = True,
    dropout_rng=None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    precision=None,
): ...

class CacheEntry(struct.PyTreeNode):
    key: np.ndarray
    value: np.ndarray
    i: np.ndarray

def multi_head_dot_product_attention(
    scope: Scope,
    inputs_q,
    inputs_kv,
    num_heads,
    dtype=...,
    qkv_features=None,
    out_features=None,
    attention_axis=None,
    causal_mask: bool = False,
    padding_mask=None,
    key_padding_mask=None,
    segmentation=None,
    key_segmentation=None,
    cache: bool = False,
    broadcast_dropout: bool = True,
    dropout_rng=None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    precision=None,
    kernel_init=...,
    bias_init=...,
    bias: bool = True,
    attention_fn=...,
): ...
def make_padding_mask(
    padding_mask_query,
    padding_mask_key,
    query_shape,
    key_shape,
    attention_axis=None,
    segmentation_mask: bool = False,
): ...
