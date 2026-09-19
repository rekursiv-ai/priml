from collections.abc import Sequence

from jax._src.lib import xla_client as xc

import numpy as np

def get_num_ways_dim_sharded(
    hlo_sharding: xc.HloSharding,
    allow_partial_manual: bool = False,
) -> tuple[list[int], int]: ...
def is_hlo_sharding_replicated(hc: xc.HloSharding) -> bool: ...
def are_hlo_shardings_equal(hc1: xc.HloSharding, hc2: xc.HloSharding) -> bool: ...
def op_sharding_to_numpy_indices(
    hlo_sharding: xc.HloSharding,
    shape: Sequence[int],
    num_devices: int,
) -> np.ndarray: ...
def op_sharding_to_indices(
    op_sharding: xc.HloSharding,
    shape: Sequence[int],
    num_devices: int,
) -> tuple[tuple[slice, ...], ...]: ...
