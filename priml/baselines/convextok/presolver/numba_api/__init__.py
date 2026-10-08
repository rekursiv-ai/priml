"""The Numba API the presolver uses; ``__init__.pyi`` narrows it for the checkers.

``typings/numba`` types a compiled kernel as a ``Dispatcher``; the presolver's
declarations type it as its Python function, and bind only its own imports.
"""

from numba import njit, prange
from numba.core.dispatcher import Dispatcher
from numba.core.types import StructRef, float64, int32, int64, uint8
from numba.experimental.structref import (
    StructRefProxy,
    define_proxy,
    new,
    register,
)
from numba.np.ufunc.parallel import get_num_threads, set_num_threads


__all__ = [
    "Dispatcher",
    "StructRef",
    "StructRefProxy",
    "define_proxy",
    "float64",
    "get_num_threads",
    "int32",
    "int64",
    "new",
    "njit",
    "prange",
    "register",
    "set_num_threads",
    "uint8",
]
