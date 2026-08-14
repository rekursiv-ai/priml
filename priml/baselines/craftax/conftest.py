"""Shared access to the reference implementation for parity tests.

The tests in this directory check the port against the JAX package it was
ported from. That package is an optional dependency, so the marker here lets
each parity test skip cleanly when it is absent.

The reference is only ever read, and only by tests. Nothing in the baseline
itself imports it, so running the environment never touches JAX.
"""

from __future__ import annotations

from typing import Any, Final

import importlib
import importlib.util
import os

import numpy as np
import pytest
import torch


# The reference package is JAX, whose CUDA backend cannot be shared: under
# ``pytest -n`` every worker tries to claim the same device and all but one
# fail to initialize a backend at all. These tests compare fixed tables and
# scalar formulas, so the CPU backend answers identically -- and portably on
# a host with no GPU or a mismatched driver. ``setdefault`` leaves an
# explicit ``JAX_PLATFORMS=cuda`` alone.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

HAS_CRAFTAX: Final = importlib.util.find_spec("craftax") is not None
"""Whether the reference implementation is installed."""

requires_craftax: Final = pytest.mark.skipif(
    not HAS_CRAFTAX,
    reason="requires the optional craftax dependency group",
)
"""Skip a parity test when the reference package is absent."""


def reference(module: str) -> Any:
    """Import a module of the reference implementation by name.

    Imported dynamically rather than at module scope so that a checkout
    without the optional dependency can still collect these tests and skip
    them, instead of failing at import.

    Args:
      module: Dotted path beneath the reference package, for example
        ``"craftax.constants"``.

    Returns:
      module: The imported reference module.

    """
    return importlib.import_module(f"craftax.{module}")


def as_tensor(array: object) -> torch.Tensor:
    """Copy a reference array into a tensor.

    The reference tables are immutable JAX arrays. Wrapping one without
    copying warns that writes would be undefined, and this suite turns
    warnings into errors.

    Args:
      array: A reference array, or anything ``numpy`` can convert.

    Returns:
      tensor: A writable copy.

    """
    return torch.from_numpy(np.array(array))
