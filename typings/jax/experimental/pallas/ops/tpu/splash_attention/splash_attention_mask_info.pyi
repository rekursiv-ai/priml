from collections.abc import Callable as Callable
from typing import NamedTuple

from _typeshed import Incomplete

import jax
import numpy as np

class MaskInfo(NamedTuple):
    data_next: np.ndarray | jax.Array | None
    mask_next: np.ndarray | jax.Array | None
    block_mask: np.ndarray | jax.Array | None
    partial_mask_blocks: np.ndarray | jax.Array | None
    q_sequence: np.ndarray | None
    is_dynamic_mask: bool = ...

class _HashableNDArray:
    array: np.ndarray
    def __init__(self, array: np.ndarray) -> None: ...
    def __hash__(self): ...
    def __eq__(self, other: object) -> bool: ...

process_mask: Incomplete
process_mask_dkv: Incomplete
process_dynamic_mask: Incomplete
process_dynamic_mask_dkv: Incomplete
