from collections.abc import Sequence

import pickle

from _typeshed import Incomplete
from jax._src.lib import xla_client as xc

import jax

def serialize(compiled: jax.stages.Compiled): ...
def deserialize_and_load(
    serialized,
    in_tree,
    out_tree,
    backend: str | xc.Client | None = None,
    execution_devices: Sequence[xc.Device] | None = None,
): ...

class _JaxPjrtPickler(pickle.Pickler):
    device_types: Incomplete
    client_types: Incomplete
    def persistent_id(self, obj): ...

class _JaxPjrtUnpickler(pickle.Unpickler):
    backend: Incomplete
    devices_by_id: Incomplete
    execution_devices: Incomplete
    def __init__(self, file, backend, execution_devices=None) -> None: ...
    def persistent_load(self, pid): ...
