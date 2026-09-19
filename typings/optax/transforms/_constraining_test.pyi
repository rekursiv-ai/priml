from optax._src import (
    combine as combine,
    transform as transform,
    update as update,
)

import chex

STEPS: int
LR: float

class ConstraintsTest(chex.TestCase):
    def test_keep_params_nonnegative(self) -> None: ...
    @chex.all_variants
    def test_zero_nans(self) -> None: ...
    def test_none_arguments(self) -> None: ...
