from _typeshed import Incomplete
from absl.testing import parameterized

import chex

class SmoothLabelsTest(parameterized.TestCase):
    ts: Incomplete
    exp_alpha_zero: Incomplete
    exp_alpha_zero_point_one: Incomplete
    exp_alpha_one: Incomplete
    def setUp(self) -> None: ...
    @chex.all_variants
    def test_scalar(self) -> None: ...
    @chex.all_variants
    def test_batched(self) -> None: ...
