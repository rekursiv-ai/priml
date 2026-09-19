import jax
import jax.numpy as jnp

def matmul_kernel(x_tile_ref, y_tile_ref, o_tile_ref, acc_ref) -> None: ...
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    block_shape,
    block_k: int = 256,
    out_dtype: jnp.dtype | None = None,
    debug: bool = False,
) -> jax.Array: ...
