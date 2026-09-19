import typing as tp

from flax.nnx import (
    extract as extract,
    graphlib as graphlib,
)
from flax.typing import (
    MISSING as MISSING,
    Missing as Missing,
)

A = tp.TypeVar("A")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])

@tp.overload
def split_inputs(*, ctxtag: str = "split_merge_inputs") -> tp.Callable[[F], F]: ...
@tp.overload
def split_inputs(f: F, *, ctxtag: str = "split_merge_inputs") -> F: ...
@tp.overload
def merge_inputs(*, ctxtag: str = "split_merge_inputs") -> tp.Callable[[F], F]: ...
@tp.overload
def merge_inputs(f: F, *, ctxtag: str = "split_merge_inputs") -> F: ...
