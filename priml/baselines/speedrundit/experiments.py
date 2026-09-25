"""Updated REG/SPRINT SpeedrunDiT training configuration."""

from __future__ import annotations

from dataclasses import field

from configgle import Makes

import torch

from priml.baselines.speedrundit.data import SpeedrunImageNetData
from priml.baselines.speedrundit.optimizers import speedrundit_optimizer
from priml.baselines.speedrundit.train_step import SpeedrunTrainStep
from priml.runtime import MultiProcess, SingleProcess
from priml.train.checkpointer import Checkpointer
from priml.train.parallelism import DataParallel, NoParallel
from priml.train.train_loop import TrainLoop


class SpeedrunTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[SpeedrunTrainStep.Config, SpeedrunImageNetData.Config],
):
    """Bind the SpeedrunDiT train step to its paired ImageNet dataset."""

    step: SpeedrunTrainStep.Config = field(default_factory=SpeedrunTrainStep.Config)
    """Model, teacher, objective, precision, and optimizer recipe."""

    dataset: SpeedrunImageNetData.Config = field(
        default_factory=SpeedrunImageNetData.Config
    )
    """Processed ImageNet and INVAE pairs."""


def exp000() -> SpeedrunTrainLoop:
    """REG/SPRINT SiT-B/1 with source Muon arithmetic and MLP scaling.

    Hypothesis:
      Reproduce the later REG branch's Muon optimizer, RMS normalization,
      value residual, contrastive flow loss, and depth-dependent MLP widths.

    References:
      https://github.com/SwayStar123/REG/tree/invae-sprint-rms-rope-valres-cfm-muon-layerwisescaling

    Results:
      TBD.

    """
    config = SpeedrunTrainLoop()
    config.study_name = "speedrundit"
    config.experiment_name = "exp000"
    config.max_steps = config.step.train_budget_steps = 400_000
    config.num_steps_eval = float(
        "inf"
    )  # The reference evaluates generated images separately.
    config.eval_every_epoch = False
    config.seed = 0
    config.runtime = MultiProcess.Config(float32_matmul_precision="high")
    config.step.parallelism = DataParallel.Config()
    assert isinstance(config.checkpointer, Checkpointer.Config)
    config.checkpointer.save_every = 10_000
    return config


def exp001() -> SpeedrunTrainLoop:
    """Keep exp000's algorithm with shared, simpler floating-point arithmetic.

    The fixed position table is computed in float32 and Muon uses the shared
    fused Newton-Schulz update. These are small numerical changes only.
    """
    config = exp000()
    config.experiment_name = "exp001"
    config.step.model.position_compute_dtype = torch.float32
    config.step.model.reference_rope = False
    config.step.optimizer = speedrundit_optimizer(reference_numerics=False)
    return config


def exp_smoke() -> SpeedrunTrainLoop:
    """Five quick single-device updates with the same training mechanisms."""
    config = exp000()
    config.experiment_name = "exp_smoke"
    config.runtime = SingleProcess.Config()
    config.step.parallelism = NoParallel.Config()
    config.max_steps = config.step.train_budget_steps = 5
    config.step.model.hidden_size = 32
    config.step.model.num_heads = 4
    config.step.model.depth = 6
    config.step.model.projector_hidden = 64
    config.step.model.projection_depths = (2, 3, 6)
    config.dataset.batch_size = 2
    config.checkpointer = None
    config.dataset.num_workers = 0
    return config
