from collections.abc import Callable
from types import FunctionType
from typing import Generic, ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")

class JITFunction(Generic[_P, _R]):
    @property
    def fn(self) -> FunctionType: ...
    @property
    def src(self) -> str: ...
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R: ...
    def __getitem__(
        self,
        grid: tuple[int, ...] | Callable[[dict[str, int]], tuple[int, ...]],
    ) -> Callable[..., object]: ...

def _normalize_ty(ty: object) -> str: ...
