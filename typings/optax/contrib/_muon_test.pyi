from _typeshed import Incomplete

import chex

UNSPECIFIED: Incomplete

def get_updates(params, muon_weight_dimension_numbers=...): ...

class MuonTest(chex.TestCase):
    def test_reshape_inverse(
        self,
        input_shape,
        dim_nums,
        expected_flat_shape,
    ) -> None: ...
    def test_callable_weight_dim_nums(self): ...
    def test_reshape_update_for_square_parameter_matches_muon_without_dim_nums(
        self,
    ) -> None: ...
    def test_reshape_and_update_single_param(self): ...
    def test_dim_nums_combinations(self): ...
