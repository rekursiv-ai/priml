from _typeshed import Incomplete

import chex

NUM_CLASSES: int
NUM_SAMPLES: int
NUM_FEATURES: int

class HessianTest(chex.TestCase):
    data: Incomplete
    labels: Incomplete
    parameters: Incomplete
    loss_fn: Incomplete
    hessian_diag: Incomplete
    def setUp(self): ...
    @chex.all_variants
    def test_hessian_diag(self) -> None: ...
