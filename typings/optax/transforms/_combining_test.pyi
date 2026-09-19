from _typeshed import Incomplete
from absl.testing import absltest
from optax._src import (
    alias as alias,
    base as base,
    transform as transform,
    update as update,
)

import chex

STEPS: int
LR: float

class CombiningTest(chex.TestCase):
    init_params: Incomplete
    per_step_updates: Incomplete
    def setUp(self) -> None: ...
    @chex.all_variants
    def test_chain(self): ...

class ExtraArgsTest(chex.TestCase):
    def test_extra_args(self): ...
    def test_extra_args_chaining(self): ...
    def test_extra_args_positional_params(self): ...

class PartitionTest(chex.TestCase):
    @chex.all_variants
    def test_partition(self, use_fn): ...
    def test_extra_args(self): ...
    def test_empty(self, container): ...
    @chex.all_variants
    def test_labels_mismatch(self, use_extra_label, use_fn): ...

def scale_by_loss(): ...

class NamedChainTest(absltest.TestCase):
    def test_named_chain(self) -> None: ...
