from __future__ import annotations

from priml.baselines.craftax.env import lead_numba
from priml.baselines.craftax.game.step import reset_range_numba
from priml.baselines.craftax.game.world_gen import (
    build_pool_numba,
    generate_world_numba,
)
from priml.baselines.craftax.ghosts.extract import _trace_numba
from priml.baselines.craftax.kernel_cache import compile_kernels
from priml.baselines.craftax.world_model.capture.step import record_rows_numba
from priml.baselines.craftax.world_model.replay import _record_numba, _run_numba


def test_each_kernel_a_cold_run_compiled_first_is_compiled() -> None:
    compile_kernels()
    for kernel in (
        generate_world_numba,
        _record_numba,
        _run_numba,
        _trace_numba,
        lead_numba,
        build_pool_numba,
        reset_range_numba,
    ):
        assert kernel.signatures, kernel.py_func.__name__
    # The default capture rules and the previous-action rules type apart.
    assert len(record_rows_numba.signatures) >= 2


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
