import typing as tp

from flax.linen import module as nn_module
from flax.nnx import (
    graphlib as graphlib,
    rnglib as rnglib,
)
from flax.nnx.bridge import wrappers as wrappers

import flax.nnx.module as nnx_module

def nnx_in_bridge_mdl(
    factory: tp.Callable[[rnglib.Rngs], nnx_module.Module],
    name: str | None = None,
) -> nnx_module.Module: ...
def linen_in_bridge_mdl(
    linen_module: nn_module.Module,
    name: str | None = None,
) -> nnx_module.Module: ...
