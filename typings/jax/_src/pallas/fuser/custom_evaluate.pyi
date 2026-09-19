import dataclasses

from _typeshed import Incomplete
from jax import lax as lax
from jax._src import (
    core as core,
    source_info_util as source_info_util,
    tree_util as tree_util,
    util as util,
)
from jax._src.pallas.fuser import fuser_utils as fuser_utils

@dataclasses.dataclass
class CustomEvaluateSettings:
    allow_transpose: bool = ...

def evaluate(f, *, allow_transpose: bool = True): ...

disallowed_primitives: Incomplete
