from typing import Any, Protocol

import chex

class MaxFun(Protocol):
    def __call__(self, scores, *args, **kwargs: Any) -> chex.Numeric: ...

def make_fenchel_young_loss(max_fun: MaxFun): ...
