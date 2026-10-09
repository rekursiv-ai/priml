from __future__ import annotations

from priml.baselines.craftax.env import lead_numba, serve_numba
from priml.baselines.craftax.game.step import reset_range_numba
from priml.baselines.craftax.game.world_gen import (
    build_pool_numba,
    generate_world_numba,
)
from priml.baselines.craftax.ghosts.extract import _trace_numba
from priml.baselines.craftax.kernel_cache import compile_kernels
from priml.baselines.craftax.world_model.capture.step import record_rows_numba
from priml.baselines.craftax.world_model.replay import _record_numba, _run_numba


# Its time is loading the cached kernels, as any first use of them in a process
# costs (0.2-0.3 s on x86); the warmup compiled them before the session's tests.
def test_each_kernel_a_cold_run_compiled_first_is_compiled() -> None:
    compile_kernels()
    for kernel in (
        generate_world_numba,
        _record_numba,
        _run_numba,
        _trace_numba,
        build_pool_numba,
    ):
        assert kernel.signatures, kernel.py_func.__name__
    # PufferLib's rules and original Craftax's type apart, as do the default
    # capture rules and the previous-action rules. ``follow_numba`` compiles as
    # ``lead_numba``'s callee and runs from Python only when a helper is parked,
    # so a process that loads ``lead_numba`` need not load it.
    for kernel in (lead_numba, serve_numba, reset_range_numba, record_rows_numba):
        assert len(kernel.signatures) >= 2, kernel.py_func.__name__


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
