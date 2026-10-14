from collections.abc import Callable
from typing import ParamSpec, TypeVar

from triton import (
    compiler as compiler,
    language as language,
    runtime as runtime,
    testing as testing,
)
from triton.runtime.jit import JITFunction as JITFunction

_P = ParamSpec("_P")
_R = TypeVar("_R")

__version__: str

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
