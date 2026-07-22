"""Package-level pytest fixtures for priml.

Clears leaked process-global runtime state between tests so runtime-touching
tests are order-independent. The monorepo's root conftest provides the same
fixture; this copy ships with the public export, which does not carry the
root conftest.
"""

from __future__ import annotations

from collections.abc import Generator

import sys

import pytest


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
