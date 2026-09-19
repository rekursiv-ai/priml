from optax._src import (
    base as base,
    linear_algebra as linear_algebra,
    numerics as numerics,
)

import chex
import jax

def clip(max_delta: chex.Numeric) -> base.GradientTransformation: ...
def clip_by_block_rms(threshold: float) -> base.GradientTransformation: ...
def clip_by_global_norm(max_norm: float) -> base.GradientTransformation: ...
def per_example_global_norm_clip(
    grads: chex.ArrayTree,
    l2_norm_clip: float,
) -> tuple[chex.ArrayTree, jax.Array]: ...
def per_example_layer_norm_clip(
    grads: chex.ArrayTree,
    global_l2_norm_clip: float,
    uniform: bool = True,
) -> tuple[chex.ArrayTree, chex.ArrayTree]: ...
def unitwise_norm(
    x: chex.Array,
    axis: int | tuple[int, ...] | None = None,
) -> chex.Array: ...
def unitwise_clip(
    g_norm: chex.Array,
    max_norm: chex.Array,
    grad: chex.Array,
    div_eps: float = 1e-06,
) -> chex.Array: ...
def adaptive_grad_clip(
    clipping: float,
    eps: float = 0.001,
    axis: int | tuple[int, ...] | None = None,
) -> base.GradientTransformation: ...
