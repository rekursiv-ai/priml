"""Native Craftax test helpers and optional-reference availability."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import copy
import functools
import importlib
import importlib.util
import os

import numpy as np
import pytest
import torch

from priml.baselines.craftax.game import world_gen


if TYPE_CHECKING:
    from types import ModuleType

    from priml.baselines.craftax.game.state import EnvState


os.environ.setdefault("JAX_PLATFORMS", "cpu")  # noqa: TID251 -- test-only backend setting, not a provisioned cache path.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # noqa: TID251 -- test-only display setting, not a provisioned cache path.

_REFERENCE_PROBES: Final = (
    "craftax.craftax.constants",
    "craftax.craftax_classic.envs.craftax_state",
)


def _reference_is_installed() -> bool:
    """Whether every reference root used by native tests is importable."""
    try:
        return all(
            importlib.util.find_spec(name) is not None for name in _REFERENCE_PROBES
        )
    except ModuleNotFoundError:
        return False


HAS_CRAFTAX: Final = _reference_is_installed()
requires_craftax: Final = pytest.mark.skipif(
    not HAS_CRAFTAX,
    reason="requires the optional craftax dependency group",
)


@functools.cache
def reference(module: str) -> ModuleType:
    """Import a module of the optional reference implementation by name."""
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
    """Return a freshly-copied generated world, generating each shape once."""
    return copy.deepcopy(_generated(num_envs, seed))


def as_tensor(array: object) -> torch.Tensor:
    """Copy a reference array into a writable tensor."""
    return torch.from_numpy(np.array(array))
