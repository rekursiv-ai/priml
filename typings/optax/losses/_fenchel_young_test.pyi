import chex
import jax.numpy as jnp

def one_hot_argmax(inputs: jnp.ndarray) -> jnp.ndarray: ...

class FenchelYoungTest(chex.TestCase):
    @chex.all_variants
    def test_fenchel_young_reg(self) -> None: ...
