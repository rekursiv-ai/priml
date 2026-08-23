"""Shared access to the reference implementation for parity tests.

The tests in this directory check the port against the JAX package it was
ported from. That package is an optional dependency, so the marker here lets
each parity test skip cleanly when it is absent.

The reference is only ever read, and only by tests. Nothing in the baseline
itself imports it, so running the environment never touches JAX.
"""

from __future__ import annotations

from typing import Any, Final

import copy
import functools
import importlib
import importlib.util
import os

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.craftax.game import world_gen
from priml.baselines.craftax.game.state import EnvState


# The reference package is JAX, whose CUDA backend cannot be shared: under
# ``pytest -n`` every worker tries to claim the same device and all but one
# fail to initialize a backend at all. These tests compare fixed tables and
# scalar formulas, so the CPU backend answers identically -- and portably on
# a host with no GPU or a mismatched driver. ``setdefault`` leaves an
# explicit ``JAX_PLATFORMS=cuda`` alone.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# The viewer under game/render is the only thing in this repo that imports
# pygame, and SDL resolves its video driver during the FIRST init of any
# subsystem, caching that choice for the process. So a test reaching pygame on
# a machine with a display opens a real window -- under xdist, one per worker,
# flickering on the operator's screen. Set here rather than in game/render:
# conftest runs at collection, before any test module imports pygame, and a
# guard at the point of use is already too late once a fixture has called
# ``pygame.init()``. ``setdefault`` leaves an explicit
# ``SDL_VIDEODRIVER=x11`` alone, for a human who wants to watch.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

_REFERENCE_PROBES: Final = (
    "craftax.craftax.constants",
    "craftax.craftax_classic.envs.craftax_state",
)
"""One real module per reference root the parity tests import.

Submodules, not the bare ``craftax`` package: an interrupted uninstall leaves
the package directory behind with its contents gone, and a root probe calls
that installed. The tests then run and fail at import instead of skipping.
"""


def _reference_is_installed() -> bool:
    """Whether every reference root the parity tests read is importable.

    ``find_spec`` returns ``None`` for a missing leaf but RAISES
    ``ModuleNotFoundError`` when an intermediate parent is gone -- which is the
    wholly-absent case this guard exists to answer, so it cannot propagate.
    """
    try:
        return all(
            importlib.util.find_spec(name) is not None for name in _REFERENCE_PROBES
        )
    except ModuleNotFoundError:
        return False


HAS_CRAFTAX: Final = _reference_is_installed()
"""Whether the reference implementation is installed."""

requires_craftax: Final = pytest.mark.skipif(
    not HAS_CRAFTAX,
    reason="requires the optional craftax dependency group",
)
"""Skip a parity test when the reference package is absent."""


@functools.cache
def reference(module: str) -> Any:
    """Import a module of the reference implementation by name.

    Imported dynamically rather than at module scope so that a checkout
    without the optional dependency can still collect these tests and skip
    them, instead of failing at import.

    Cached because the first import builds JAX's world-generation tables and
    costs seconds, while every later one is a dict lookup. Uncached, pytest
    charges that whole cost to whichever parity test happened to run first,
    which reads as a slow test rather than a slow import -- and under xdist
    every worker pays it again.

    Args:
      module: Dotted path beneath the reference package, for example
        ``"constants"``.

    Returns:
      module: The imported reference module.

    """
    return importlib.import_module(f"craftax.{module}")


@functools.cache
def _generated(num_envs: int, seed: int) -> EnvState:
    """Generate one world and keep it for the process."""
    return world_gen.generate_world(
        num_envs=num_envs,
        generator=torch.Generator().manual_seed(seed),
        device=torch.device("cpu"),
    )


def generated_world(*, num_envs: int = 1, seed: int = 0) -> EnvState:
    """Return a freshly-copied generated world, generating each shape once.

    Generating costs ~25 ms and does NOT get cheaper with fewer workers: it is
    a thousand small tensor ops dispatched from Python, so the cost is flat in
    the batch size. Copying one costs 0.85 ms. Memoizing by (workers, seed)
    and handing out deep copies therefore removes almost all of the bill while
    every caller still gets a world it may mutate freely.

    Args:
      num_envs: Parallel worlds in the batch.
      seed: World seed.

    Returns:
      state: A private copy of the cached world.

    """
    return copy.deepcopy(_generated(num_envs, seed))


def as_tensor(array: object) -> Tensor:
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
