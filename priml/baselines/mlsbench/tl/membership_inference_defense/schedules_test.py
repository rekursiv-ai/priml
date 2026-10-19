"""Tests for the step decay, against torch's ``MultiStepLR``."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.mlsbench.tl.membership_inference_defense.schedules import (
    multistep,
)


@pytest.mark.parametrize(
    "milestones",
    [[150, 225], [2, 2, 5], [5, 2]],
    ids=["recipe", "repeated", "unsorted"],
)
def test_multistep_is_torchs_recurrence(milestones: list[int]) -> None:
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=0.1,
    )
    for epoch in range(240):
        lr = multistep(
            epoch,
            base_lr=0.1,
            total_epochs=240,
            milestones=milestones,
            gamma=0.1,
        )
        assert lr == optimizer.param_groups[0]["lr"], epoch
        optimizer.step()
        scheduler.step()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
