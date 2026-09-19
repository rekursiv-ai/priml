from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

from flax.linen.module import (
    Module as Module,
    compact as compact,
)

class Sequential(Module):
    layers: Sequence[Callable[..., Any]]
    def __post_init__(self) -> None: ...
    @compact
    def __call__(self, *args, **kwargs): ...
