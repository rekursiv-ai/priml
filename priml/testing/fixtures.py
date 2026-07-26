"""Testing utilities."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import importlib
import sys
import warnings

import pytest


if TYPE_CHECKING:
    import torch as torch_typed


try:
    import torch
except ImportError:  # torch is optional; cleanup_cuda + get_device no-op without it.
    torch = None  # ty: ignore[invalid-assignment] -- optional dep sentinel; pyright infers Module|None


def get_device() -> torch_typed.device:
    """Return the preferred test device, CUDA when available.

    Returns:
      device: ``cuda`` if a CUDA device is present, else ``cpu``.

    Raises:
      RuntimeError: If torch is not installed.

    """
    if torch is None:
        raise RuntimeError("torch is not installed")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def cleanup_cuda() -> Generator[None]:
    """Reclaim CUDA memory symmetrically around each test (no-op on CPU)."""
    _reclaim_cuda()
    yield
    _reclaim_cuda()


@contextmanager
def torch_compiler_isolation() -> Generator[None]:
    """Isolate torch.compile process-global state around a block of code.

    Wrap any block whose code calls ``torch.compile``. On exit it resets Dynamo
    and clears Inductor's caches, so nothing downstream reuses the block's
    guards or codegen. Blocks that never import Dynamo skip the reset entirely.

    Entry also warms one upstream import; see
    ``_warm_inductor_mkldnn_import`` for why that is load-bearing under
    warnings-as-errors.

    Yields:
      None: Control returns to the wrapped block.

    """
    _warm_inductor_mkldnn_import()
    try:
        yield
    finally:
        # Skip when the block never compiled: plain ``import torch`` pulls in
        # neither Dynamo nor Inductor, so most tests avoid loading Inductor at
        # all here. An ``if`` rather than an early ``return`` -- a ``return``
        # inside ``finally`` would discard an in-flight exception.
        if torch is not None and "torch._dynamo" in sys.modules:
            from torch._inductor.utils import clear_caches  # noqa: PLC0415 -- lazy

            torch._dynamo.reset()  # noqa: SLF001 -- documented cache-clear entrypoint
            clear_caches()


def _reclaim_cuda() -> None:
    """Synchronize and empty the CUDA cache when a CUDA device is present."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _warm_inductor_mkldnn_import() -> None:
    """Import the module Inductor loads lazily, before warnings turn fatal.

    Inductor's pre-grad pass imports ``torch.utils.mkldnn``, whose module body
    calls the deprecated ``torch.jit.script_method`` on Python 3.14
    (pytorch/pytorch#127283). Under the suite's warnings-as-errors that import
    raises, unwinds itself back out of ``sys.modules``, and so raises again on
    every later compile or ``torch._dynamo.reset()``. Importing it once here,
    successfully, means the warning is never raised again in this process --
    so wrapped blocks run with warnings-as-errors fully armed rather than
    inside a suppression window.

    Costs ~4ms on the first call, then ~0.1us. Remove once torch drops the
    ``script_method`` call from that module.
    """
    # Callers warm on every wrapped block, so short-circuit once the module is
    # resident: re-entering ``catch_warnings`` bumps the global filter version
    # and invalidates every module's warning registry, ~2.4us a call.
    if torch is None or "torch.utils.mkldnn" in sys.modules:
        return
    # ``import torch.utils.mkldnn`` would bind the top-level name ``torch``,
    # turning the sentinel checked above into a local (UnboundLocalError). The
    # ``from`` form avoids that but needs F401 plus PLC0415 for a name we never
    # use, so import for the side effect alone and bind nothing.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*torch\.jit\.script_method.*",
            category=DeprecationWarning,
        )
        importlib.import_module("torch.utils.mkldnn")
