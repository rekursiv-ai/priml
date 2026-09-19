from contextlib import contextmanager
from typing import Any, Generic, NoReturn, overload

from _typeshed import Incomplete

class Config:
    flax_use_flaxlib: bool
    flax_array_ref: bool
    flax_pytree_module: bool
    flax_max_repr_depth: int | None
    flax_always_shard_variable: bool
    flax_hijax_variable: bool
    nnx_graph_mode: bool
    def __init__(self) -> None: ...
    @overload
    def update(self, name: str, value: Any, /) -> None: ...
    @overload
    def update(self, holder: FlagHolder[_T], value: _T, /) -> None: ...
    @contextmanager
    def temp_flip_flag(self, var_name: str, var_value: bool): ...

config: Incomplete

class FlagHolder(Generic[_T]):
    name: Incomplete
    __doc__: Incomplete
    def __init__(self, name, help) -> None: ...
    def __bool__(self) -> NoReturn: ...
    @property
    def value(self) -> _T: ...

def bool_flag(name: str, *, default: bool, help: str) -> FlagHolder[bool]: ...
def int_flag(name: str, *, default: int | None, help: str) -> FlagHolder[int]: ...
def static_bool_env(varname: str, default: bool) -> bool: ...
def static_int_env(varname: str, default: int | None) -> int | None: ...

flax_filter_frames: Incomplete
flax_profile: Incomplete
flax_use_orbax_checkpointing: Incomplete
flax_preserve_adopted_names: Incomplete
flax_return_frozendict: Incomplete
flax_fix_rng: Incomplete
flax_use_flaxlib: Incomplete
flax_array_ref: Incomplete
flax_pytree_module: Incomplete
flax_max_repr_depth: Incomplete
flax_always_shard_variable: Incomplete
flax_hijax_variable: Incomplete
nnx_graph_mode: Incomplete
