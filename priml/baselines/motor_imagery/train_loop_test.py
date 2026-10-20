"""The full-cohort loop's declared absence of evaluation."""

from __future__ import annotations

from typing import NoReturn, override

import torch

from priml.baselines.motor_imagery.data import Trial, WindowData
from priml.baselines.motor_imagery.experiments import exp_smoke
from priml.baselines.motor_imagery.model import MultiScaleEEGNet
from priml.baselines.motor_imagery.train_loop import MotorImageryLoop
from priml.baselines.motor_imagery.train_step import MotorImageryStep
from priml.runtime import SingleProcess
from priml.train.parallelism import NoParallel


class UnevaluatedData(WindowData):
    """Fail if the loop asks for held-out windows."""

    @override
    def eval_dataloader(self) -> NoReturn:
        raise AssertionError("The full-cohort loop must not evaluate.")


def test_declared_full_cohort_loop_trains_without_evaluating() -> None:
    campaign = exp_smoke()
    model = campaign.ensemble_loop.step.model = MultiScaleEEGNet.Config(8, 2)
    model.num_samples = 100
    model.channels_hidden = model.channels_projection = 2
    campaign.ensemble_loop.runtime = SingleProcess.Config(device="cpu")
    campaign.ensemble_loop.step.parallelism = NoParallel.Config(device="cpu")
    campaign.ensemble_loop.dataset.batch_size = 4
    cfg = campaign.finalized().ensemble_loop
    data = UnevaluatedData(
        cfg.dataset,
        trials=[
            Trial(subject="S001", trial_id=0, label=0, windows=torch.zeros(4, 8, 100)),
        ],
    )
    loop = MotorImageryLoop(cfg, step=cfg.step.make(), dataset=data)
    loop.train()
    assert isinstance(loop.step, MotorImageryStep)
    assert loop.step.global_step == loop.dataset.timer_epoch.global_count == 3
    assert loop.step.timer_eval.global_count == 0


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
