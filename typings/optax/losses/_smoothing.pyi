import chex
import jax.numpy as jnp

def smooth_labels(labels: chex.Array, alpha: float) -> jnp.ndarray: ...
