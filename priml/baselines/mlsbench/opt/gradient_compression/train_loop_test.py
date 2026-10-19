"""Tests for the compressing loop; the evaluation policy is tested in ``dl/vision``."""

from __future__ import annotations

from priml.baselines.mlsbench.dl.vision.train_loop import EvalEpochsTrainLoop
from priml.baselines.mlsbench.opt.gradient_compression.experiments import exp000
from priml.baselines.mlsbench.opt.gradient_compression.train_loop import (
    CompressionTrainLoop,
)
from priml.baselines.mlsbench.opt.gradient_compression.train_step import (
    CompressedTrainStep,
)


def test_the_loop_builds_itself_on_the_compressing_step() -> None:
    config = exp000()
    assert isinstance(config.step, CompressedTrainStep.Config)
    assert CompressionTrainLoop.Config.parent_class is CompressionTrainLoop
    assert issubclass(CompressionTrainLoop, EvalEpochsTrainLoop)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
