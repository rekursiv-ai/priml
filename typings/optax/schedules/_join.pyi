from collections.abc import Sequence

from optax._src import base as base

def join_schedules(
    schedules: Sequence[base.Schedule],
    boundaries: Sequence[int],
) -> base.Schedule: ...
