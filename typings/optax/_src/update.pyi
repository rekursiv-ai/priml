from optax._src.base import Params, Updates

def apply_updates(params: Params, updates: Updates) -> Params: ...
def incremental_update(
    new_tensors: Params,
    old_tensors: Params,
    step_size: float,
) -> Params: ...
def periodic_update(
    new_tensors: Params,
    old_tensors: Params,
    steps: int,
    update_period: int,
) -> Params: ...
