from typing import TypedDict

import dataclasses

from absl.testing import absltest
from optax._src import (
    alias as alias,
    base as base,
    combine as combine,
    transform as transform,
)

import chex

@dataclasses.dataclass
class FakeShardSpec:
    sharding_axis: int | None

class ScaleByAdamStateDict(TypedDict):
    count: chex.Array
    params: None

class StateUtilsTest(absltest.TestCase):
    def test_dict_based_optimizers(self): ...
    def test_state_chex_dataclass(self): ...
    def test_adam(self): ...
    def test_inject_hparams(self): ...
    def test_map_params_to_none(self) -> None: ...
    def test_map_non_params_to_none(self): ...
    def test_tree_get_all_with_path(self): ...
    def test_tree_get(self): ...
    def test_tree_set(self): ...
