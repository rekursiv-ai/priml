"""EEG state records, source-name adapters, and the exp000 step recipe for tests."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, cast

import re

from torch import Tensor, nn

import torch

from priml.baselines.motor_imagery.data import Preprocessing
from priml.baselines.motor_imagery.experiments import exp000
from priml.baselines.motor_imagery.model import SOURCE_NAMES
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.lr_schedule import RecursiveCosine
from priml.testing.golden import mismatches, read_tensors
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from configgle import Makeable

    from priml.baselines.motor_imagery.train_step import MotorImageryStep
    from priml.testing.golden import PinnedRandom


def current_key(key: str) -> str:
    """Name a source record key with the blessed nouns the networks now use.

    Args:
      key: A record key, optionally under a prefix such as ``final/``.

    Returns:
      key: The key with each attribute in ``SOURCE_NAMES`` renamed.

    """
    return re.sub(
        rf"(?<!\w)({'|'.join(SOURCE_NAMES)})(?=\.)",
        lambda match: SOURCE_NAMES[match[1]],
        key,
    )


def read_source_record(path: Path) -> dict[str, Tensor]:
    """Read a source-minted record, naming its tensors with the current state keys."""
    return {current_key(key): value for key, value in read_tensors(path).items()}


def model_record(model: nn.Module, *, prefix: str) -> dict[str, Tensor]:
    """Capture every model tensor without storing a hash or truncated view."""
    return {
        f"{prefix}/{name}": value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def optimizer_record(optimizer: torch.optim.Optimizer) -> dict[str, Tensor]:
    """Capture all optimizer tensor state in stable parameter traversal order.

    Args:
      optimizer: Reference-compatible AdamW optimizer.

    Returns:
      record: Every per-parameter optimizer tensor.

    """
    if isinstance(optimizer, CompositeOptimizer):
        if len(optimizer.optimizers) != 1:
            raise ValueError("Expected len(optimizer.optimizers) == 1.")
        optimizer = optimizer.optimizers[0]
    states = cast("dict[Tensor, dict[str, Tensor | float | int]]", optimizer.state)
    return {
        f"optimizer/{index}/{name}": value.detach().cpu().clone()
        for index, state in enumerate(states.values())
        for name, value in state.items()
        if isinstance(value, Tensor)
    }


def random_record(pins: PinnedRandom) -> dict[str, Tensor]:
    """Capture primitive counts; missing or additional call sites fail replay."""
    return {f"random/{name}": torch.tensor(count) for name, count in pins.calls.items()}


def compare_record(
    expected: Mapping[str, Tensor],
    actual: Mapping[str, Tensor],
) -> None:
    """Compare every stored bit and report all differing fields together.

    Args:
      expected: Complete source-minted record.
      actual: Complete native replay record.

    Raises:
      ValueError: Any key, shape, dtype or stored bit differs.

    """
    differences = mismatches(
        _initializer_sites(expected),
        _initializer_sites(actual),
        ulps=True,
    )
    if differences:
        raise ValueError("\n".join(differences))


def source_preprocessing() -> Preprocessing.Config:
    """Build exp000's filters and windows; each call returns a fresh tree."""
    preprocessing = exp000().validation.dataset.preprocessing
    assert isinstance(preprocessing, Preprocessing.Config)
    return preprocessing


def source_step(model: Makeable[nn.Module], *, epochs: int) -> MotorImageryStep.Config:
    """Build exp000's step recipe on the CPU, annealing over ``epochs`` data passes.

    Args:
      model: Network the step trains.
      epochs: Complete passes the budget and cosine horizon cover.

    Returns:
      step: A fresh tree, so a test may edit it.

    """
    step = exp000().validation.step
    step.model = model
    step.parallelism = NoParallel.Config(device="cpu")
    step.train_budget_epochs = epochs
    schedule = step.lr_schedule
    assert isinstance(schedule, RecursiveCosine.Config)
    schedule.T_max = epochs
    return step


# Priml overrides torch's reset methods. Their file line numbers differ while the
# weight/bias draw sites retain their order, shape, dtype, and full values. Keep every
# distinct site and its count; this only removes source line numbers.
def _initializer_sites(record: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Name each initializer site by its encounter position within its layer type."""
    counts: Counter[str] = Counter()
    normalized: dict[str, Tensor] = {}
    for key, value in record.items():
        name = key
        if key.startswith("random/aten.uniform_."):
            group = key.split(":")[0]
            name = f"{group}:{counts[group]}"
            counts[group] += 1
        normalized[name] = value
    return normalized
