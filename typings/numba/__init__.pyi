# A whole stub package, not an overlay of numba's own inline stubs: a package
# directory without this file is a namespace that basedpyright resolves per
# import cache, so `from numba.core import config` read this tree in one run and
# numba's untyped `config.py` in another.
from collections.abc import Callable, Mapping
from typing import (
    Literal,
    Protocol,
    TypedDict,
    TypeVar,
    Unpack,
    overload,
    type_check_only,
)

from numba.core.ccallback import CFunc
from numba.core.dispatcher import Dispatcher
from numba.core.types import Type
from numba.core.typing.templates import Signature
from numba.np.ufunc.parallel import (
    get_num_threads as get_num_threads,
    set_num_threads as set_num_threads,
)

__version__: str

_F = TypeVar("_F", bound=Callable[..., object])

@type_check_only
class _JITOptions(TypedDict, total=False):
    nogil: bool
    parallel: bool
    fastmath: bool | set[str]
    error_model: Literal["python", "numpy"]
    looplift: bool
    forceinline: bool
    debug: bool

@type_check_only
class _JITWrapper(Protocol):
    def __call__(self, fn: _F, /) -> Dispatcher[_F]: ...

@type_check_only
class _CFuncWrapper(Protocol):
    def __call__(self, fn: _F, /) -> CFunc[_F]: ...

@overload
def njit(
    signature_or_function: None = None,
    *,
    locals: Mapping[str, Type | str] = ...,
    cache: bool = False,
    boundscheck: bool | None = None,
    **options: Unpack[_JITOptions],
) -> _JITWrapper: ...
@overload
def njit(
    signature_or_function: _F,
    *,
    locals: Mapping[str, Type | str] = ...,
    cache: bool = False,
    boundscheck: bool | None = None,
    **options: Unpack[_JITOptions],
) -> Dispatcher[_F]: ...
def cfunc(
    sig: str | Signature,
    *,
    locals: Mapping[str, Type | str] = ...,
    cache: bool = False,
    **options: Unpack[_JITOptions],
) -> _CFuncWrapper: ...

# A class whose `__new__` returns `range`; in a parallel kernel the compiler
# splits the loop, and in Python it is `range`.
def prange(start: int, stop: int = ..., step: int = ..., /) -> range: ...
