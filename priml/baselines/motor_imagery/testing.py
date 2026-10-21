"""EEG state records and the exp000 preprocessing and step recipes for tests."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

import torch

from priml.baselines.motor_imagery.data import Preprocessing
from priml.baselines.motor_imagery.experiments import exp000
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.lr_schedule import RecursiveCosine
from priml.testing.golden import mismatches
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Mapping

    from configgle import Makeable

    from priml.baselines.motor_imagery.train_step import MotorImageryStep
    from priml.testing.golden import PinnedRandom


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
    """Count each primitive's call sites, numbered in encounter order per file.

    A site's line number moves with unrelated edits to its file; its position
    among that file's sites does not, so a missing or added site still fails.

    Args:
      pins: The pinned sampler after the replayed work.

    Returns:
      record: One count per ``random/<primitive>/<file>:<index>`` site.

    """
    counts: Counter[str] = Counter()
    record: dict[str, Tensor] = {}
    for name, count in pins.calls.items():
        site = name.rsplit(":", 1)[0]
        record[f"random/{site}:{counts[site]}"] = torch.tensor(count)
        counts[site] += 1
    return record


def compare_record(
    expected: Mapping[str, Tensor],
    actual: Mapping[str, Tensor],
) -> None:
    """Compare every stored bit and report all differing fields together.

    Args:
      expected: Complete expected record.
      actual: Complete replayed record.

    Raises:
      ValueError: Any key, shape, dtype or stored bit differs.

    """
    differences = mismatches(expected, actual, ulps=True)
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
