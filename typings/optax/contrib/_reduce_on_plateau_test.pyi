from _typeshed import Incomplete
from absl.testing import parameterized

class ReduceLROnPlateauTest(parameterized.TestCase):
    patience: int
    cooldown: int
    transform: Incomplete
    updates: Incomplete
    def setUp(self) -> None: ...
    def tearDown(self) -> None: ...
    def test_learning_rate_reduced_after_cooldown_period_is_over(
        self,
        enable_x64,
    ) -> None: ...
    def test_learning_rate_is_not_reduced(self, enable_x64) -> None: ...
    def test_learning_rate_not_reduced_during_cooldown(self, enable_x64) -> None: ...
    def test_learning_rate_not_reduced_after_end_scale_is_reached(
        self,
        enable_x64,
    ) -> None: ...
