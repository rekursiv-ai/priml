from optax import contrib as contrib
from optax._src import (
    alias as alias,
    base as base,
    combine as combine,
    numerics as numerics,
    update as update,
    utils as utils,
)

import chex

class ContribTest(chex.TestCase):
    def test_optimizers_accept_extra_args(
        self,
        opt_name,
        opt_kwargs,
        wrapper_name,
        wrapper_kwargs,
        wrap,
    ): ...
    def test_optimizers(
        self,
        opt_name,
        opt_kwargs,
        wrapper_name,
        wrapper_kwargs,
        target,
        dtype,
    ): ...
    @chex.all_variants
    def test_optimizers_can_be_wrapped_in_inject_hyperparams(
        self,
        opt_name,
        opt_kwargs,
        wrapper_name=None,
        wrapper_kwargs=None,
    ): ...
    def test_preserve_dtype(
        self,
        opt_name,
        opt_kwargs,
        dtype,
        wrapper_name=None,
        wrapper_kwargs=None,
    ): ...
    def test_gradient_accumulation(
        self,
        opt_name,
        opt_kwargs,
        dtype,
        wrapper_name=None,
        wrapper_kwargs=None,
    ): ...
    def test_state_shape_dtype_shard_stability(
        self,
        opt_name,
        opt_kwargs,
        wrapper_name,
        wrapper_kwargs,
        dtype,
    ): ...
    def test_optimizers_accept_learning_rate_schedule_if_type_annotated_as_such(
        self,
        opt_name,
        opt_kwargs,
        wrapper_name,
        wrapper_kwargs,
    ): ...
