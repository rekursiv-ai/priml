from flax.core import Scope as Scope
from flax.linen import initializers as initializers

def batch_norm(
    scope: Scope,
    x,
    use_running_average: bool = False,
    axis: int = -1,
    momentum: float = 0.99,
    epsilon: float = 1e-05,
    dtype=...,
    bias: bool = True,
    scale: bool = True,
    bias_init=...,
    scale_init=...,
    axis_name=None,
    axis_index_groups=None,
    kind: str = "batch_stats",
): ...
def layer_norm(
    scope: Scope,
    x,
    epsilon: float = 1e-06,
    dtype=...,
    bias: bool = True,
    scale: bool = True,
    bias_init=...,
    scale_init=...,
): ...
def group_norm(
    scope,
    x,
    num_groups: int = 32,
    group_size=None,
    epsilon: float = 1e-06,
    dtype=...,
    bias: bool = True,
    scale: bool = True,
    bias_init=...,
    scale_init=...,
): ...
