from collections.abc import Callable
from typing import Generic, ParamSpec, TypeVar, overload

from numba.core.types import Type
from numba.core.typing.templates import Signature

_F_co = TypeVar("_F_co", bound=Callable[..., object], covariant=True)
_P = ParamSpec("_P")
_R = TypeVar("_R")

class Dispatcher(Generic[_F_co]):
    # The keyword arguments the decorator compiled with, `nopython` included.
    targetoptions: dict[str, object]
    @property
    def py_func(self) -> _F_co: ...
    @property
    def signatures(self) -> list[tuple[Type, ...]]: ...
    def __call__(
        self: Dispatcher[Callable[_P, _R]],
        /,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R: ...
    def compile(
        self: Dispatcher[Callable[_P, _R]],
        sig: str | tuple[Type, ...] | Signature,
    ) -> Callable[_P, _R]: ...
    def enable_caching(self) -> None: ...
    @overload
    def inspect_llvm(self, signature: None = None) -> dict[tuple[Type, ...], str]: ...
    @overload
    def inspect_llvm(self, signature: tuple[Type, ...]) -> str: ...
    @overload
    def inspect_asm(self, signature: None = None) -> dict[tuple[Type, ...], str]: ...
    @overload
    def inspect_asm(self, signature: tuple[Type, ...]) -> str: ...
