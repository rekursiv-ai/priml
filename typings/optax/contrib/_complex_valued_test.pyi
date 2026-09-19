from optax._src import (
    transform as transform,
    update as update,
)

import chex

class ComplexValuedTest(chex.TestCase):
    @chex.all_variants
    def test_split_real_and_imaginary(self, scaler_constr): ...
