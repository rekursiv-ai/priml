"""The vision evaluation-policy loop, narrowed to the compressing step."""

from __future__ import annotations

from dataclasses import field

from configgle import Makes

from priml.baselines.mlsbench.dl.vision.data import Cifar100Data
from priml.baselines.mlsbench.dl.vision.train_loop import EvalEpochsTrainLoop
from priml.baselines.mlsbench.opt.gradient_compression.train_step import (
    CompressedTrainStep,
)


class CompressionTrainLoop(EvalEpochsTrainLoop):
    """A loop that evaluates after the epochs its policy picks, stepping with compression."""

    class Config(
        Makes["CompressionTrainLoop"],
        EvalEpochsTrainLoop.Config[CompressedTrainStep.Config, Cifar100Data.Config],
    ):
        """The compressing step, prepared CIFAR tensors and the evaluation policy."""

        step: CompressedTrainStep.Config = field(
            default_factory=CompressedTrainStep.Config,
        )
        """Model, optimizer, loss, schedule and compressor."""
