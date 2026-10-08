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

from contextlib import ExitStack
from typing import TYPE_CHECKING, Protocol, cast

import random
import sys

import pytest

from priml.lib.testing import userdirs_fixture
from priml.lib.testing.resource_markers import pytest_collection_modifyitems
from priml.lib.testing.threads import cap_math_threads
from priml.lib.testing.userdirs_fixture import isolate_user_dirs
from priml.testing import regenerate
from priml.testing.fixtures import cleanup_cuda


if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from priml.distributed.testing import WarmPoolGetter


class _BitGenerator(Protocol):
    state: object


class _NumpyRandom(Protocol):
    def seed(self, value: int) -> None: ...
    def PCG64(self, value: int) -> _BitGenerator: ...  # noqa: N802 -- NumPy exposes this generator with its public name.


class _NumpyModule(Protocol):
    random: _NumpyRandom


class _NumpyGenerator(Protocol):
    bit_generator: _BitGenerator


class _TorchModule(Protocol):
    def manual_seed(self, value: int) -> object: ...


# Re-exported, not merely imported: an autouse fixture reaches only the
# directory of the conftest that names it, so binding it here is what points
# every priml test's XDG lookups at a tmp dir instead of the developer's own,
# and what reclaims CUDA memory around every priml test instead of only the
# files that remembered to import ``cleanup_cuda``.
__all__ = [
    "cleanup_cuda",
    "isolate_user_dirs",
    "pytest_addoption",
    "pytest_collection_modifyitems",
    "pytest_configure",
    "seed_rng",
]


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--regenerate-golden`` and ``--regenerate-b4b``.

    Defined here so the exported package ships the flags; the repo-root
    conftest calls this rather than registering its own.
    """
    regenerate.add_options(parser)


def pytest_configure(config: pytest.Config) -> None:
    """Declare the ``real_user_dirs`` marker and read the regeneration flags."""
    userdirs_fixture.pytest_configure(config)
    regenerate.configure(config)


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
    process, not just priml's -- ``TrainLoop`` is constructed by other
    packages' suites too. Public (not ``_``-prefixed) so the
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
        setattr(runtime, "_runtime_initialized", False)  # noqa: B010 -- The module is a ``sys.modules`` lookup, so the attribute has no static type.


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
    numpy = cast(_NumpyModule | None, sys.modules.get("numpy"))
    if numpy is not None:
        numpy.random.seed(1337)
        # ``priml.math.seed.numpy_rng`` is the house Generator; it is a
        # separate stream from the legacy module RNG seeded above.
        seed = sys.modules.get("priml.math.seed")
        if seed is not None:
            rng = cast(_NumpyGenerator, seed.numpy_rng)
            rng.bit_generator.state = numpy.random.PCG64(1337).state
    torch = cast(_TorchModule | None, sys.modules.get("torch"))
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
    from priml.distributed.testing import (  # noqa: PLC0415 -- Keeps torch.distributed off collection of non-distributed tests.
        WorkerPool,
    )

    pools: dict[tuple[tuple[str, int], ...], WorkerPool] = {}
    with ExitStack() as stack:

        def get_pool(mesh_dims: Mapping[str, int]) -> WorkerPool:
            key = tuple(mesh_dims.items())
            if key not in pools:
                made = WorkerPool.Config(mesh_dims=dict(mesh_dims)).make()
                pools[key] = stack.enter_context(made)
            return pools[key]

        yield get_pool
