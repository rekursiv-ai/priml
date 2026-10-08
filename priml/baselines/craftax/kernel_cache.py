"""Fill Numba's cache with the game kernels the tests reach first.

A kernel compiles on its first call for each signature its arguments' types
fix, unless Numba's cache already holds it (``game/jit.py``). Each call in
:func:`compile_kernels` is one the tests make, so each kernel lands in the cache
with the signature the tests use, and every later process loads it.
``conftest.py`` runs it once per pytest session, before any test, as

  python -m priml.baselines.craftax.kernel_cache
"""

from __future__ import annotations

from priml.baselines.craftax.ghosts.extract import extract
from priml.baselines.craftax.testing import tiny_env
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.capture.env import CaptureEnv, Schedule


def compile_kernels() -> None:
    """Compile, or load, each kernel a cold unit test run compiled for over a second.

    A recorded episode compiles world generation and ``replay._record_numba``;
    extracting it, ``replay._run_numba`` and ``extract._trace_numba``; a step
    of :func:`~priml.baselines.craftax.testing.tiny_env`, ``lead_numba``
    and the pool and reset kernels; a buffer step under each capture rule set,
    ``record_rows_numba``, whose previous-action rules type apart.
    """
    episode = replay.record(world_seed=1, sampling_seed=1, max_decisions=20_000)
    extract(episode, ordinal=0)
    env = tiny_env().make()
    try:
        env.reset()
        env.step_buffer(0)
    finally:
        env.close()
    for previous_action in (False, True):
        config = CaptureEnv.Config(num_envs=2)
        config.rules.previous_action = previous_action
        CaptureEnv(config, schedule=Schedule(budget=1_000)).step_buffer(0)


if __name__ == "__main__":
    compile_kernels()
