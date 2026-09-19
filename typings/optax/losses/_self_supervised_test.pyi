from _typeshed import Incomplete
from absl.testing import parameterized

import chex

class NtxentTest(chex.TestCase):
    ys: Incomplete
    ys_2: Incomplete
    ts_1: Incomplete
    ts_2: Incomplete
    exp_1: Incomplete
    exp_2: Incomplete
    exp_3: Incomplete
    def setUp(self) -> None: ...
    @chex.all_variants
    def test_batched(self) -> None: ...

class TripletMarginLossTest(chex.TestCase, parameterized.TestCase):
    a1: Incomplete
    p1: Incomplete
    n1: Incomplete
    a2: Incomplete
    p2: Incomplete
    n2: Incomplete
    def setUp(self) -> None: ...
    @chex.all_variants
    def test_batched(self, anchor, positive, negative, margin): ...
    @chex.all_variants
    def test_vmap(self, anchor, positive, negative) -> None: ...
