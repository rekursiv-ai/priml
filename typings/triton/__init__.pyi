from collections.abc import Callable
from typing import Generic, ParamSpec, TypeVar

from triton import (
    compiler as compiler,
    language as language,
    runtime as runtime,
    testing as testing,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")

__version__: str

class JITFunction(Generic[_P, _R]):
    @property
    def src(self) -> str: ...
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R: ...
    def __getitem__(
        self,
        grid: tuple[int, ...] | Callable[[dict[str, int]], tuple[int, ...]],
    ) -> Callable[..., object]: ...

class Config:
    def __init__(
        self,
        kwargs: dict[str, int],
        *,
        num_warps: int = ...,
        num_stages: int = ...,
    ) -> None: ...

class Autotuner:
    def __getitem__(
        self,
        grid: tuple[int, ...] | Callable[[dict[str, int]], tuple[int, ...]],
    ) -> Callable[..., object]: ...

def autotune(
    *,
    configs: list[Config],
    key: list[str],
) -> Callable[[JITFunction[_P, _R]], Autotuner]: ...
def jit(fn: Callable[_P, _R]) -> JITFunction[_P, _R]: ...
def cdiv(x: int, y: int) -> int: ...
def next_power_of_2(n: int) -> int: ...
