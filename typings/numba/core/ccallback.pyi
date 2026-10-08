from collections.abc import Callable
from typing import Generic, ParamSpec, TypeVar

_F_co = TypeVar("_F_co", bound=Callable[..., object], covariant=True)
_P = ParamSpec("_P")
_R = TypeVar("_R")

class CFunc(Generic[_F_co]):
    def __call__(
        self: CFunc[Callable[_P, _R]],
        /,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R: ...
    @property
    def address(self) -> int: ...
