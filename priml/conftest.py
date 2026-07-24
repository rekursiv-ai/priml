"""Package-level pytest fixtures for priml.

Clears leaked process-global runtime state between tests so runtime-touching
tests are order-independent, and provides the ``warm_pools`` fixture the
distributed integration tests need. Both are defined here (not only in the
monorepo root conftest) so the public export -- which ships this file but not
the root conftest -- can still collect and run those tests: everything
``warm_pools`` needs lives inside priml (``priml.distributed.testing``), so the
fixture is self-contained in the exported package.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import ExitStack
from typing import TYPE_CHECKING

import sys

import pytest


if TYPE_CHECKING:
    from priml.distributed.testing import WarmPoolGetter


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
