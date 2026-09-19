from _typeshed import Incomplete

import chex

STEPS: int

class AddingTest(chex.TestCase):
    init_params: Incomplete
    per_step_updates: Incomplete
    def setUp(self) -> None: ...
    @chex.all_variants
    def test_add_decayed_weights(self) -> None: ...
    @chex.all_variants
    def test_add_noise_has_correct_variance_scaling(self): ...
    def test_none_argument(self) -> None: ...
