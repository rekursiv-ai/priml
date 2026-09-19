from typing import NamedTuple

from _typeshed import Incomplete
from optax._src import (
    base as base,
    update as update,
)

import chex

class OptimizerTestState(NamedTuple):
    aggregate_grads: base.Params

class MechanicTest(chex.TestCase):
    grads: Incomplete
    initial_params: Incomplete
    def setUp(self) -> None: ...
    def loop(self, optimizer, num_steps, params): ...
    def test_mechanized(self) -> None: ...
