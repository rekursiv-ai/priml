from collections.abc import Callable as Callable

from jax._src.linear_util import WrappedFun as WrappedFun

def wrap_init(f: Callable, params=None, *, debug_info=None) -> WrappedFun: ...
