from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

import numpy as np

device_kind_handler_dict: dict[str, Callable[..., np.ndarray | None]]

def create_device_mesh(
    mesh_shape: Sequence[int],
    devices: Sequence[Any] | None = None,
    *,
    contiguous_submeshes: bool = False,
    allow_split_physical_axes: bool = False,
) -> np.ndarray: ...
def create_hybrid_device_mesh(
    mesh_shape: Sequence[int],
    dcn_mesh_shape: Sequence[int],
    devices: Sequence[Any] | None = None,
    *,
    process_is_granule: bool = False,
    should_sort_granules_by_key: bool = True,
    allow_split_physical_axes: bool = False,
) -> np.ndarray: ...
