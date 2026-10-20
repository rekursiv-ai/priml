"""Typed EEG fold configuration."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path

from configgle import Makes

from priml.baselines.motor_imagery.data import WindowData
from priml.baselines.motor_imagery.train_step import MotorImageryStep
from priml.train.train_loop import TrainLoop


class MotorImageryLoop(TrainLoop):
    """Narrow the loop to the reference step and windowed dataset."""

    class Config(
        Makes["MotorImageryLoop"],
        TrainLoop.Config[MotorImageryStep.Config, WindowData.Config],
    ):
        step: MotorImageryStep.Config = field(default_factory=MotorImageryStep.Config)
        """Reference objective and epoch-indexed optimizer recipe."""

        dataset: WindowData.Config = field(default_factory=WindowData.Config)
        """Reconstructible raw inputs, preprocessing, fold people, and batching."""

        working_dir: Path | str = "/runs/{study_name}/{experiment_name}/fits"
        """Logical fit artifact directory, beside the campaign receipt."""
