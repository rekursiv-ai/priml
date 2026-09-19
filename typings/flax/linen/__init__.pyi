from flax.linen.activation import (
    gelu as gelu,
    relu as relu,
    sigmoid as sigmoid,
    silu as silu,
    swish as swish,
    tanh as tanh,
)
from flax.linen.attention import (
    MultiHeadDotProductAttention as MultiHeadDotProductAttention,
    SelfAttention as SelfAttention,
)
from flax.linen.linear import (
    Dense as Dense,
    Embed as Embed,
)
from flax.linen.module import (
    Module as Module,
    compact as compact,
    nowrap as nowrap,
)
from flax.linen.normalization import (
    BatchNorm as BatchNorm,
    GroupNorm as GroupNorm,
    LayerNorm as LayerNorm,
    RMSNorm as RMSNorm,
)
from flax.linen.spmd import (
    logical_to_mesh as logical_to_mesh,
    logical_to_mesh_axes as logical_to_mesh_axes,
    logical_to_mesh_sharding as logical_to_mesh_sharding,
    with_logical_constraint as with_logical_constraint,
)
from flax.linen.stochastic import Dropout as Dropout
from flax.linen.summary import tabulate as tabulate
from flax.linen.transforms import add_metadata_axis as add_metadata_axis
