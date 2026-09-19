import typing as tp

from flax.typing import Initializer as Initializer

type DtypeLikeInexact = tp.Any

def zeros_init() -> Initializer: ...
def ones_init() -> Initializer: ...
