from absl.testing import parameterized
from optax._src import (
    alias as alias,
    base as base,
    combine as combine,
    transform as transform,
    update as update,
)

import chex

class ConditionalityTest(parameterized.TestCase):
    def test_apply_if_finite(self, opt_builder): ...
    def test_apply_if_finite_pmap(self): ...

class ConditionallyTransformTest(chex.TestCase):
    NUM_STEPS: int
    @chex.all_variants
    def test_stateless_inner(self): ...
    @chex.all_variants
    def test_statefull_inner(self): ...

class ConditionallyMaskTest(chex.TestCase):
    NUM_STEPS: int
    MIN_LOSS: float
    @chex.all_variants
    def test_stateless_inner(self): ...
    @chex.all_variants
    def test_statefull_inner(self): ...
    @chex.all_variants
    def test_stateless_inner_with_extra_args(self): ...
