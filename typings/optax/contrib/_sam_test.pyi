from optax._src import (
    alias as alias,
    combine as combine,
    numerics as numerics,
    update as update,
)

import chex

class SAMTest(chex.TestCase):
    def test_optimization(
        self,
        base_opt_name,
        base_opt_kwargs,
        adv_opt_name,
        adv_opt_kwargs,
        sync_period,
        target,
        dtype,
        opaque_mode,
    ): ...
