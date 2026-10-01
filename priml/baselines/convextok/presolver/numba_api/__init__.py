"""The Numba API the presolver uses; ``__init__.pyi`` types it for the checkers.

Numba ships no type stubs. A package-level stub in ``typings/`` would retype
every Numba user in the repository, several of which rely on Numba being
untyped, so the declarations bind only the presolver's imports.
"""

from numba import njit, prange
from numba.core.dispatcher import Dispatcher
from numba.core.types import StructRef, float64, int32, int64, uint8
from numba.experimental.structref import (
    StructRefProxy,
    define_proxy,  # pyright: ignore[reportUnknownVariableType] -- Numba is untyped; __init__.pyi types it.
    new,  # pyright: ignore[reportUnknownVariableType] -- Numba is untyped; __init__.pyi types it.
    register,  # pyright: ignore[reportUnknownVariableType] -- Numba is untyped; __init__.pyi types it.
)


__all__ = [
    "Dispatcher",
    "StructRef",
    "StructRefProxy",
    "define_proxy",
    "float64",
    "int32",
    "int64",
    "new",
    "njit",
    "prange",
    "register",
    "uint8",
]
