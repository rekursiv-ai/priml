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


if TYPE_CHECKING:
    from priml.distributed.testing import WarmPoolGetter


def cap_math_threads() -> None:
    """Give each process one math thread, before any native library loads.

    xdist parallelizes at the process level, so a worker that also spawns a
    full-width BLAS/OpenMP pool oversubscribes the box: N workers x N threads.
    The effect is not a mild slowdown -- an 8-worker run turns 2ms training
    steps into seconds and trips per-test timeouts.

    Must run before NumPy/PyTorch/SciPy import: torch reads ``OMP_NUM_THREADS``
    at import and pins its ATen intra-op pool to match. Conftest import is early
    enough; ``addopts`` is not.

    ``MKL_CBWR`` pins a CPU-independent GEMM kernel so float32 matmul is
    bit-identical across x86 hosts (Intel AVX-512 vs AMD EPYC), which
    numeric-parity tests compare under tight tolerances. MKL reads it only at
    its first GEMM, so it too must be set before any matmul runs.

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
    os.environ.setdefault("MKL_CBWR", "COMPATIBLE")


cap_math_threads()


@pytest.fixture(autouse=True)
def _reset_runtime_global() -> Generator[None]:  # pyright: ignore[reportUnusedFunction] -- pytest invokes autouse fixtures by injection, not by name
    """Clear a leaked process-global runtime flag after every test.

    ``priml.runtime`` keeps a module-global ``_runtime_initialized``. A
    test that initializes the real runtime and fails to tear it down leaves the
    flag set, so a later test sharing the same worker process sees a dirty
    runtime and its own ``initialize()`` raises ``RuntimeError: Runtime already
    initialized``. Clearing the flag at teardown makes runtime-touching tests
    order-independent.

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
