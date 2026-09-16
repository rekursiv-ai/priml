"""Testing utilities."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import sys

import pytest


if TYPE_CHECKING:
    import torch
else:
    from wrapt import lazy_import

    # Deferred, not `try: import torch`: the repo-root conftest re-exports
    # ``cleanup_cuda`` through ``priml.conftest``, so an eager import here
    # would load torch at collection time for every package in the repo.
    torch = lazy_import("torch")


def get_device() -> torch.device:
    """Return the preferred test device, CUDA when available.

    Returns:
      device: ``cuda`` if a CUDA device is present, else ``cpu``.

    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def cleanup_cuda() -> Generator[None]:
    """Reclaim CUDA memory symmetrically around each test (no-op on CPU).

    Only acts when torch is already imported: a test that never loaded torch
    has no CUDA cache to reclaim, and importing it here would make torch a
    dependency of every test the autouse scope reaches.

    Teardown runs BEFORE ``monkeypatch`` undoes the test's patches, so a test
    that fakes CUDA state is still faked here. Reading a seam the test controls
    is therefore a live hazard, and so is trusting a stale setup-time snapshot
    over it -- the gate below reads both.

    Yields:
      item: Each yielded value.

    """
    torch_loaded_at_setup = "torch" in sys.modules
    on_cuda_host = torch_loaded_at_setup and torch.cuda.is_available()
    if on_cuda_host and _cuda_is_initialized():
        # Same gate as teardown: with no context there is nothing to reclaim,
        # and reaching `synchronize` would force the lazy init anyway.
        _reclaim_cuda()
    yield
    if torch_loaded_at_setup:
        # All three are load-bearing: the setup snapshot, because a test faking
        # `is_available()` True on a CPU-only host must not reach `synchronize`;
        # `_cuda_is_initialized()`, because forcing CUDA's lazy init on a host
        # the test never touched can raise; and a live `is_available()`, because
        # a test that faked CUDA away is still faked when this runs.
        if on_cuda_host and _cuda_is_initialized() and torch.cuda.is_available():
            _reclaim_cuda()
    elif _cuda_is_initialized():
        # Torch first imported by the test body: no setup snapshot either way,
        # so reclaim only if that import path actually created a context.
        _reclaim_cuda()


@contextmanager
def torch_compiler_isolation() -> Generator[None]:
    """Isolate torch.compile process-global state around a block of code.

    Wrap any block whose code calls ``torch.compile``. On exit it resets Dynamo
    and clears Inductor's caches, so nothing downstream reuses the block's
    guards or codegen. Blocks that never import Dynamo skip the reset entirely.

    Yields:
      None: Control returns to the wrapped block.

    """
    try:
        yield
    finally:
        # Skip when the block never compiled: plain ``import torch`` pulls in
        # neither Dynamo nor Inductor, so most tests avoid loading Inductor at
        # all here. An ``if`` rather than an early ``return`` -- a ``return``
        # inside ``finally`` would discard an in-flight exception.
        if "torch._dynamo" in sys.modules:
            from torch._inductor.utils import (  # noqa: PLC0415 -- Test fixtures load optional training dependencies only when requested.
                clear_caches,
            )

            torch._dynamo.reset()  # noqa: SLF001 -- The fixture resets the private runtime state under test.
            clear_caches()


def poison_free_pool(*shapes: tuple[int, ...], blocks: int = 32) -> None:
    """Leave NaN in freed heap blocks that a later ``torch.empty`` may reuse.

    A module that allocates parameters with ``torch.empty`` and defers filling
    them to an external initializer hands its caller whatever the allocator had
    left in that block. A unit test that builds such a module directly then
    passes or fails on what previous tests happened to free, not on the module.
    Poisoning first makes that dependence visible: a parameter the module leaves
    unwritten reads back NaN instead of plausible-looking numbers.

    Best effort by construction -- which block a platform's allocator returns is
    its own business, so a caller must assert that values it *does* define are
    correct, never that unwritten memory is observably poisoned. Filling many
    blocks per shape rather than one keeps this effective on allocators that do
    not reuse most-recently-freed blocks first.

    Args:
      shapes: Every parameter shape the module under test allocates. The
        allocator pools by block size, so a shape that is not poisoned can
        still come back clean.
      blocks: How many blocks to fill and free per shape.

    """
    for shape in shapes:
        # Built and dropped one at a time rather than held in a list: each block
        # has to be freed before the next same-size allocation can receive it.
        for _ in range(blocks):
            block = torch.full(shape, float("nan"))
            del block


def _cuda_is_initialized() -> bool:
    """Report whether a real CUDA context exists in this process."""
    return "torch" in sys.modules and torch.cuda.is_initialized()


def _reclaim_cuda() -> None:
    """Synchronize and empty the CUDA cache. Callers gate on CUDA being present."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
