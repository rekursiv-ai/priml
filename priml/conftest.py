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
import random
import sys

import pytest

from priml.lib.testing.resource_markers import pytest_collection_modifyitems
from priml.lib.testing.userdirs_fixture import (
    isolate_user_dirs,
    pytest_configure,
)


if TYPE_CHECKING:
    from priml.distributed.testing import WarmPoolGetter


# Re-exported, not merely imported: an autouse fixture reaches only the
# directory of the conftest that names it, so binding it here is what points
# every priml test's XDG lookups at a tmp dir instead of the developer's own.
__all__ = [
    "isolate_user_dirs",
    "pytest_collection_modifyitems",
    "pytest_configure",
    "seed_rng",
]


def cap_math_threads() -> None:
    """Give each process one math thread, before any native library loads.

    xdist parallelizes at the process level, so a worker that also spawns a
    full-width BLAS/OpenMP pool oversubscribes the box: N workers x N threads.
    The effect is not a mild slowdown -- an 8-worker run turns 2ms training
    steps into seconds and trips per-test timeouts.

    Must run before NumPy/PyTorch/SciPy import: torch reads ``OMP_NUM_THREADS``
    at import and pins its ATen intra-op pool to match. Conftest import is early
    enough; ``addopts`` is not.

    ``MKL_CBWR`` pins a CPU-independent GEMM kernel. It is NOT redundant with
    the bfb harness's float64 upcast: that upcast removes the float32 kernel's
    error, but a float64 GEMM's reduction order still varies with the kernel
    MKL selects, and a float64 difference lands on a different float32 bit
    whenever the exact value sits near a rounding boundary. Absorbed almost
    always, not always -- which is a test that fails on one machine in many,
    the worst failure a golden can have. Removing it was measured inert on an
    AMD host, where MKL takes a generic path anyway, and broke an Intel one.
    MKL reads it at its first GEMM, so it must be set before any matmul runs.

    Every variable uses ``setdefault``, so an explicit
    ``OMP_NUM_THREADS=8 pytest`` always wins.
    """
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",  # macOS Accelerate.
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")  # noqa: TID251 -- test/env knob, not a provisioned cache path
    os.environ.setdefault("MKL_CBWR", "COMPATIBLE")  # noqa: TID251 -- test/env knob, not a provisioned cache path


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


@pytest.fixture(autouse=True)
def seed_rng() -> None:
    """Start every test from one RNG state.

    Python's ``random``, NumPy, and torch RNGs are process globals, so under
    xdist a test's draws depend on which tests its worker ran before it.
    Reseeding at entry makes leaked state unobservable; nothing needs
    restoring afterwards. A test wanting a specific stream still seeds
    itself.

    Each library is seeded only if it is already imported: a process that
    never loaded torch has no torch RNG to leak, and importing it here would
    make torch a test dependency of every package in the repo. Reading
    ``sys.modules`` rather than importing keeps that true.
    """
    random.seed(1337)
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        numpy.random.seed(1337)
        # ``priml.math.seed.numpy_rng`` is the house Generator; it is a
        # separate stream from the legacy module RNG seeded above.
        seed = sys.modules.get("priml.math.seed")
        if seed is not None:
            seed.numpy_rng.bit_generator.state = numpy.random.PCG64(1337).state
    torch = sys.modules.get("torch")
    if torch is not None:
        # Seeds CPU and every visible CUDA device (through its lazy-init
        # path when no context exists yet), as ``set_seed_local`` does.
        torch.manual_seed(1337)


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
