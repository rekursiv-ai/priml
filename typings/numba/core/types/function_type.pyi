from collections.abc import Collection, Iterable, Sequence
from typing import (
    Any,
    ClassVar,
    Final,
    Never,
    Protocol,
    Self,
    override,
    runtime_checkable,
    type_check_only,
)
from typing_extensions import TypeVar

import abc

from numba.core.compiler import CompileResult
from numba.core.dispatcher import Dispatcher
from numba.core.types.abstract import Type
from numba.core.typing.context import Context
from numba.core.typing.templates import Signature

__all__ = [
    "CompileResultWAP",
    "FunctionPrototype",
    "FunctionType",
    "UndefinedFunctionType",
    "WrapperAddressProtocol",
]

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)

@type_check_only
class _CanUnliteral(Protocol[_T_co]):
    def __unliteral__(self, /) -> _T_co: ...

@type_check_only
class _HasLiteralType(Protocol[_T_co]):
    @property
    def literal_type(self, /) -> _T_co: ...

# represents the return type of `types.unliteral` for literals
type _LiteralLike[_T] = _CanUnliteral[_T] | _HasLiteralType[_T]

###

class FunctionType(Type):
    cconv: ClassVar[None] = None

    nargs: Final[int]
    signature: Final[Signature]
    ftype: Final[FunctionPrototype]
    _key: str

    def __init__(self, signature: Signature | _LiteralLike[Signature]) -> None: ...

    #
    @property
    @override
    def key(self) -> str: ...
    @property  # type: ignore[misc]
    @override
    def name(self) -> str: ...  # pyrefly:ignore[bad-override]

    #
    @override
    def is_precise(self) -> bool: ...
    def get_precise(self) -> FunctionType: ...
    @override
    def dump(self, tab: str = "") -> None: ...
    def get_call_type(
        self,
        context: Context,
        args: Sequence[Type],
        kws: dict[Never, Never] | None,  # empty dict or None
    ) -> Signature: ...
    def check_signature(self, other_sig: Signature) -> bool: ...
    @override
    def unify(self, typingctx: Context, other: Type) -> Self | None: ...

class UndefinedFunctionType(FunctionType):
    dispatchers: Final[Collection[Dispatcher]]

    def __init__(self, nargs: int, dispatchers: Collection[Dispatcher]) -> None: ...

class FunctionPrototype(Type):
    cconv: ClassVar[None] = None
    rtype: Type
    atypes: tuple[Type, ...]

    def __init__(self, rtype: Type, atypes: Iterable[Type]) -> None: ...

    #
    @property
    @override
    def key(self) -> str: ...

@runtime_checkable
class WrapperAddressProtocol(Protocol):
    @abc.abstractmethod
    def __wrapper_address__(self) -> int: ...
    @abc.abstractmethod
    def signature(self) -> Signature: ...

class CompileResultWAP(WrapperAddressProtocol):
    def __init__(self, cres: CompileResult) -> None: ...
    def dump(self, tab: str = "") -> None: ...

    #
    @override
    def __wrapper_address__(self) -> int: ...
    @override
    def signature(self) -> Signature: ...

    #
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
