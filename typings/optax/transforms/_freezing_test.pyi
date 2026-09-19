from _typeshed import Incomplete
from absl.testing import parameterized
from optax._src import (
    alias as alias,
    base as base,
    update as update,
)

PARAMS_FLAT: Incomplete
PARAMS_NESTED: Incomplete
GRAD_FLAT: Incomplete
GRAD_NESTED: Incomplete

class FreezeTest(parameterized.TestCase):
    def test_freeze_updates(
        self,
        params,
        grads,
        freeze_mask,
        expected_updates,
    ) -> None: ...
    def test_nested_freeze_all(self): ...
    def test_nested_freeze_none(self) -> None: ...
    def test_scalar_bool_broadcast(self, scalar_mask) -> None: ...
    def test_bad_structure_raises(self) -> None: ...
    def test_partial_prefix_mask_behavior(self) -> None: ...

class SelectiveTransformTest(parameterized.TestCase):
    def test_selective_transform_effect(
        self,
        params,
        grads,
        mask,
        expected_params,
    ) -> None: ...
    def test_nested_train_all(self): ...
    def test_scalar_freeze_all(self, scalar_mask) -> None: ...
    def test_selective_bad_structure(self) -> None: ...
    def test_partial_prefix_mask_behavior(self) -> None: ...
