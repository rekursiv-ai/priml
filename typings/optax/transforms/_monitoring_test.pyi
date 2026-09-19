from absl.testing import absltest
from optax import tree as tree
from optax._src import (
    alias as alias,
    update as update,
)

class HooksTest(absltest.TestCase):
    def test_snapshot(self): ...
