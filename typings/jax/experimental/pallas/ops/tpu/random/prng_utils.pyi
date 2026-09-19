from collections.abc import Sequence

from _typeshed import Incomplete
from jax import lax as lax

type Shape = Sequence[int]
round_up: Incomplete

def blocked_iota(block_shape: Shape, total_shape: Shape): ...
def compute_scalar_offset(iteration_index, total_size: Shape, block_size: Shape): ...
