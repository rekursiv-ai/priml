from collections.abc import Sequence

from _typeshed import Incomplete
from jax.experimental import mesh_utils as mesh_utils

import jax

Device: Incomplete

class TopologyDescription:
    devices: list[Device]
    def __init__(self, devices: list[Device]) -> None: ...

def get_attached_topology(platform=None) -> TopologyDescription: ...
def get_topology_desc(
    topology_name: str = "",
    platform: str | None = None,
    **kwargs,
) -> TopologyDescription: ...
def make_mesh(
    topo: TopologyDescription,
    mesh_shape: Sequence[int],
    axis_names: tuple[str, ...],
    *,
    contiguous_submeshes: bool = False,
) -> jax.sharding.Mesh: ...
