"""Pin indexed cosine recurrence to the complete optimizer-rate sequence."""

from __future__ import annotations

import math

import pytest
import torch

from priml.optimizers.lr_schedule import RecursiveCosine


@pytest.mark.parametrize("horizon", [3, 200, 300])
@pytest.mark.parametrize("minimum", [0.0, 5e-7])
def test_indexed_cosine_matches_all_torch_rates(horizon: int, minimum: float) -> None:
    optimizer = torch.optim.SGD(
        [
            {"params": [torch.nn.Parameter(torch.zeros(2, 3))], "lr": 0.0002},
            {"params": [torch.nn.Parameter(torch.zeros(3))], "lr": 0.00007},
        ],
    )
    reference = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=horizon,
        eta_min=minimum,
    )
    expected = [reference.get_last_lr()]
    for _ in range(3 * horizon + 2):
        optimizer.step()
        reference.step()
        expected.append(reference.get_last_lr())

    config = RecursiveCosine.Config()
    config.T_max = horizon
    config.eta_min = minimum
    schedule = config.make()
    for index in [*range(len(expected)), 2, 0, horizon, 2]:
        for group, initial in enumerate((0.0002, 0.00007)):
            actual_rate = float(schedule.learning_rate(initial, step=index))
            expected_rate = float(expected[index][group])
            assert actual_rate.hex() == expected_rate.hex()
    fresh = config.make()
    assert fresh.learning_rate(0.0002, step=horizon - 1) == expected[horizon - 1][0]


@pytest.mark.parametrize("horizon", [0, -3])
def test_cosine_rejects_nonpositive_horizon(horizon: int) -> None:
    config = RecursiveCosine.Config()
    config.T_max = horizon
    with pytest.raises(ValueError, match="T_max"):
        config.make()


@pytest.mark.parametrize("minimum", [math.nan, math.inf, -math.inf])
def test_cosine_rejects_nonfinite_floor(minimum: float) -> None:
    config = RecursiveCosine.Config()
    config.eta_min = minimum
    with pytest.raises(ValueError, match="eta_min"):
        config.make()


def test_cosine_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match=r"^step must be nonnegative\.$"):
        RecursiveCosine.Config().make().learning_rate(0.0002, step=-1)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
