from collections.abc import Callable

from optax._src import base as base

import chex
import jax

class Normal:
    def sample(
        self,
        key: jax.typing.ArrayLike,
        sample_shape: base.Shape = (),
        dtype: jax.typing.DTypeLike = ...,
    ) -> jax.Array: ...
    def log_prob(self, inputs: jax.Array) -> jax.Array: ...

class Gumbel:
    def sample(
        self,
        key: jax.typing.ArrayLike,
        sample_shape: base.Shape = (),
        dtype: jax.typing.DTypeLike = ...,
    ) -> jax.Array: ...
    def log_prob(self, inputs: jax.Array) -> jax.Array: ...

def make_perturbed_fun(
    fun: Callable[[chex.ArrayTree], chex.ArrayTree],
    num_samples: int = 1000,
    sigma: float = 0.1,
    noise=...,
    use_baseline: bool = True,
) -> Callable[[chex.PRNGKey, chex.ArrayTree], chex.ArrayTree]: ...
