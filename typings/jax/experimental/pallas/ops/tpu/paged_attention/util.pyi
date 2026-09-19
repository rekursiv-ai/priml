from _typeshed import Incomplete
from jax.experimental.pallas.ops.tpu.paged_attention import (
    quantization_utils as quantization_utils,
)

import jax

MASK_VALUE: Incomplete

def grouped_query_attention_reference(
    queries: jax.Array,
    k_pages: jax.Array,
    v_pages: jax.Array,
    seq_lens: jax.Array,
    soft_cap: float | None = None,
    debug: bool = False,
) -> jax.Array: ...
