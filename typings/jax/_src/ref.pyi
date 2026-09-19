from typing import Any

from jax._src import core as core

def new_ref(init_val: Any, *, memory_space: Any = None) -> core.Ref: ...
