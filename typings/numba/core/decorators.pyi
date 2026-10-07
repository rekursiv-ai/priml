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

from numba.core import compiler, ir, types, typing
from numba.core.ccallback import CFunc
from numba.core.dispatcher import Dispatcher
from numba.core.typing.templates import _inline_info

###
# type-check only helpers

type _FnInlineUntyped = Callable[[ir.Expr, ir.FunctionIR, ir.FunctionIR], bool]
type _FnInlineTyped = Callable[[ir.Expr, _inline_info, _inline_info], bool]
type _FnInline = _FnInlineUntyped | _FnInlineTyped

type _ToSignature = typing.Signature | tuple[types.Type | typing.Signature, ...] | str
# The generic `list` type is invariant, so we can't use `list[_ToSignature]`.
# So we instead use a "free" type variable allow the list type argument to vary.
_ToSignatureT = TypeVar("_ToSignatureT", bound=_ToSignature)

type _ToLocals = Mapping[str, types.Type | str]

_FunctionT = TypeVar("_FunctionT", bound=Callable[..., object])

@type_check_only
class _JITOptions(TypedDict, total=False):
    looplift: bool
    nogil: bool
    parallel: bool
    fastmath: bool | set[str]
    error_model: Literal["python", "numpy"]
    inline: Literal["never", "always"] | _FnInline
    forceinline: bool
    debug: bool

@type_check_only
class _JITWrapper(Protocol):
    def __call__(self, fn: _FunctionT, /) -> Dispatcher[_FunctionT]: ...

@type_check_only
class _CFuncWrapper(Protocol):
    def __call__(self, fn: _FunctionT, /) -> CFunc[_FunctionT]: ...

###
# stubs

@overload  # signature
def jit(
    signature_or_function: _ToSignature | list[_ToSignatureT] | None = None,
    locals: _ToLocals = ...,
    cache: bool = False,
    pipeline_class: type[compiler.CompilerBase] | None = None,
    boundscheck: bool | None = None,
    *,
    nopython: bool = True,
    forceobj: bool = False,
    **options: Unpack[_JITOptions],
) -> _JITWrapper: ...
@overload  # function
def jit(
    signature_or_function: _FunctionT,
    locals: _ToLocals = ...,
    cache: bool = False,
    pipeline_class: type[compiler.CompilerBase] | None = None,
    boundscheck: bool | None = None,
    *,
    nopython: bool = True,
    forceobj: bool = False,
    **options: Unpack[_JITOptions],
) -> Dispatcher[_FunctionT]: ...

#
@overload  # signature
def njit(
    signature_or_function: _ToSignature | list[_ToSignatureT] | None = None,
    locals: _ToLocals = ...,
    cache: bool = False,
    pipeline_class: type[compiler.CompilerBase] | None = None,
    boundscheck: bool | None = None,
    **options: Unpack[_JITOptions],
) -> _JITWrapper: ...
@overload  # function
def njit(
    signature_or_function: _FunctionT,
    locals: _ToLocals = ...,
    cache: bool = False,
    pipeline_class: type[compiler.CompilerBase] | None = None,
    boundscheck: bool | None = None,
    **options: Unpack[_JITOptions],
) -> Dispatcher[_FunctionT]: ...

#
def cfunc(
    sig: _ToSignature,
    locals: _ToLocals = ...,
    cache: bool = False,
    pipeline_class: type[compiler.CompilerBase] | None = None,
    **options: Unpack[_JITOptions],
) -> _CFuncWrapper: ...

#
def jit_module(
    *,
    locals: _ToLocals = ...,
    cache: bool = False,
    pipeline_class: type[compiler.CompilerBase] | None = None,
    boundscheck: bool | None = None,
    nopython: bool = True,
    forceobj: bool = False,
    **kwargs: Unpack[_JITOptions],
) -> None: ...
