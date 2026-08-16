"""Package-level pytest fixtures for priml.

Owns every piece of test setup priml needs: the native-thread caps, the
process-global runtime reset that keeps runtime-touching tests
order-independent, and the ``warm_pools`` fixture the distributed integration
tests need. All of it lives here rather than in the monorepo's repo-root
conftest, which does not ship: the exported package would otherwise lose the
caps and an 8-worker run would oversubscribe the box. The root conftest imports
from this module instead of repeating it, so the two cannot drift.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import ExitStack
from typing import TYPE_CHECKING

import os
import sys

import pytest

from priml.lib.testing.userdirs_fixture import (
    isolate_user_dirs,
    pytest_configure,
)


if TYPE_CHECKING:
    from priml.distributed.testing import WarmPoolGetter


# Re-exported, not merely imported: an autouse fixture reaches only the
# directory of the conftest that names it, so binding it here is what points
# every priml test's XDG lookups at a tmp dir instead of the developer's own.
__all__ = ["isolate_user_dirs", "pytest_configure"]


def cap_math_threads() -> None:
    """Give each process one math thread, before any native library loads.

    xdist parallelizes at the process level, so a worker that also spawns a
    full-width BLAS/OpenMP pool oversubscribes the box: N workers x N threads.
    The effect is not a mild slowdown -- an 8-worker run turns 2ms training
    steps into seconds and trips per-test timeouts.

    Must run before NumPy/PyTorch/SciPy import: torch reads ``OMP_NUM_THREADS``
    at import and pins its ATen intra-op pool to match. Conftest import is early
    enough; ``addopts`` is not.

    Scheduling only -- no numeric pin belongs here. An ``MKL_CBWR`` was carried
    to make float32 GEMM bit-identical across x86 vendors; the bfb harness now
    computes matmul in float64 and rounds once, which does that job for every
    BLAS. Measured with the pin defeated: the arcagi1 golden still replays, and
    priml, baselines, and the integration-marked parity tests are unchanged.

    Every variable uses ``setdefault``, so an explicit
    ``OMP_NUM_THREADS=8 pytest`` always wins.
    """
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",  # macOS Accelerate
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")


cap_math_threads()


@pytest.fixture(autouse=True)
def reset_runtime_global() -> Generator[None]:
    """Clear a leaked process-global runtime flag after every test.

    ``priml.runtime`` keeps a module-global ``_runtime_initialized``. A
    test that initializes the real runtime and fails to tear it down leaves the
    flag set, so a later test sharing the same worker process sees a dirty
    runtime and its own ``initialize()`` raises ``RuntimeError: Runtime already
    initialized``. Clearing the flag at teardown makes runtime-touching tests
    order-independent.

    The flag is a PROCESS global, so the guard has to cover every test in the
    process, not just priml's -- ``TrainLoop`` is constructed by suites under
    baselines/ and experimental/ too. Public (not ``_``-prefixed) so the
    repo-root conftest can re-export it and widen the autouse scope to the whole
    repo; priml keeps the definition so the public export ships it.

    Only acts if the module is already imported -- importing it eagerly in
    conftest would pull in ``torch.distributed`` at collection time.
    """
    yield
    runtime = sys.modules.get("priml.runtime")
    if runtime is not None and getattr(runtime, "_runtime_initialized", False):
        # Private-global reset: the module exposes no setter (none should exist
        # in prod); tests are the only context that may leak it.
        setattr(runtime, "_runtime_initialized", False)  # noqa: B010 -- dynamic module global, no static attr


@pytest.fixture(scope="session")
def warm_pools() -> Generator[WarmPoolGetter]:
    """Yield a getter for session-cached, reused distributed ``WorkerPool``s.

    Tests sharing a mesh shape reuse one warm pool keyed on ``mesh_dims``,
    paying the ~1.8s ``WorkerPool`` spawn once per shape per xdist worker
    instead of once per test. Reuse is safe only because every dispatched
    worker fn catches its own exceptions, reseeds RNG, and resets any
    process-global it touches.

    Yields:
      get_pool: Maps ``mesh_dims`` to a live, entered ``WorkerPool``.

    """
    # Deferred (not module scope) so collecting non-distributed tests never
    # imports torch.distributed.
    from priml.distributed.testing import WorkerPool  # noqa: PLC0415

    pools: dict[tuple[tuple[str, int], ...], WorkerPool] = {}
    with ExitStack() as stack:

        def get_pool(mesh_dims: Mapping[str, int]) -> WorkerPool:
            key = tuple(mesh_dims.items())
            if key not in pools:
                made = WorkerPool.Config(mesh_dims=dict(mesh_dims)).make()
                pools[key] = stack.enter_context(made)
            return pools[key]

        yield get_pool
