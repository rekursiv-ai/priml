"""Tests for the ``optimization-gradient-compression`` experiment chain.

``exp000`` is pinned by value and by its golden; each fork is checked as a
delta, exactly which fields differ from ``exp000``, so "one slot per idea" is a
property of the code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import inspect

import pytest
import torch

from priml.baselines.mlsbench.dl.vision.experiments_test import (
    changed_fields,
    write_dataset,
)
from priml.baselines.mlsbench.dl.vision.metrics import BestAccuracy
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.opt.gradient_compression import experiments
from priml.baselines.mlsbench.opt.gradient_compression.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
)
from priml.baselines.mlsbench.opt.gradient_compression.metrics import (
    MeanCrossEntropy,
)
from priml.baselines.mlsbench.opt.gradient_compression.model import VGG
from priml.runtime import SingleProcess
from priml.testing.golden import assert_pprint_golden
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from pathlib import Path

    from priml.baselines.mlsbench.opt.gradient_compression.train_loop import (
        CompressionTrainLoop,
    )
    from priml.testing.experiments import ExperimentFactory

ALL_EXPERIMENTS: Final[list[ExperimentFactory[CompressionTrainLoop.Config]]] = [
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
]

ERROR_FEEDBACK: Final = {
    "step.compressor",
    "step.compressor.func",
    "step.compressor.beta",
    "step.compressor.compressor",
    "step.compressor.compressor.func",
}
"""What replacing the identity with an error-feedback wrapper changes."""


def shrink(
    config: CompressionTrainLoop.Config,
    *,
    directory: Path,
) -> CompressionTrainLoop.Config:
    """Narrow ``config`` to a size a CPU test can run, preserving its recipe.

    Only SIZE changes: depth, width, batch, horizon. Optimizer, schedule, loss,
    compressor and augmentation stay exactly as the experiment set them.
    """
    assert isinstance(config.runtime, SingleProcess.Config)
    config.runtime.device = "cpu"
    config.step.parallelism = NoParallel.Config(device="cpu")
    model = config.step.model
    if isinstance(model, ResNet.Config):
        model.blocks_per_stage = (1, 1, 1)
        model.channels_hidden = (4, 5, 6)
    else:
        assert isinstance(model, VGG.Config)
        model.channels_hidden = ((4,), (5, 6))
        model.channels_head = (7, 8)
    config.step.train_budget_epochs = config.max_epochs = 1
    config.base_dir = None
    config.working_dir = directory
    config.dataset.working_dir = directory
    config.dataset.batch_size = config.dataset.eval_batch_size = 4
    config.dataset.translate_pad = 1
    config.checkpointer = None
    config.tracker = None
    return config


@pytest.mark.parametrize(
    ("factory", "changed"),
    [
        (exp001, {*ERROR_FEEDBACK, "step.compressor.compressor.ratio"}),
        (
            exp002,
            {
                "step.compressor.func",
                "step.compressor.levels",
                "step.compressor.clip_norm",
            },
        ),
        (exp003, ERROR_FEEDBACK),
    ],
    ids=["exp001", "exp002", "exp003"],
)
def test_a_baseline_changes_only_the_compressor(
    factory: ExperimentFactory[CompressionTrainLoop.Config],
    changed: set[str],
) -> None:
    assert changed_fields(exp000(), factory()) == {"experiment_name", *changed}


def test_exp004_changes_only_the_network_data_and_rate() -> None:
    changed = changed_fields(exp000(), exp004())
    assert {f for f in changed if not f.startswith("step.model")} == {
        "experiment_name",
        "step.optimizer.lr",
        "dataset.working_dir",
        *(f"dataset.{name}[{i}]" for name in ("mean", "std") for i in range(3)),
    }
    assert "step.model" in changed


def test_exp005_changes_only_the_depth() -> None:
    assert changed_fields(exp000(), exp005()) == {
        "experiment_name",
        *(f"step.model.blocks_per_stage[{i}]" for i in range(3)),
    }


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_experiment_trains_at_minimum_size(
    factory: ExperimentFactory[CompressionTrainLoop.Config],
    tmp_path: Path,
) -> None:
    write_dataset(tmp_path, classes=10)
    loop = shrink(factory(), directory=tmp_path).make()
    loop.train()
    assert loop.step.global_step == 2
    assert factory().experiment_name == factory.__name__


def test_exp000_pins_the_reference_recipe() -> None:
    cfg = exp000()
    assert isinstance(cfg.step.model, ResNet.Config)
    assert cfg.step.model.blocks_per_stage == (3, 3, 3)
    assert cfg.step.model.channels_out == 10
    step = cfg.step.make()
    assert isinstance(step.optimizer, torch.optim.SGD)
    group = step.optimizer.param_groups[0]
    assert (group["lr"], group["momentum"], group["weight_decay"]) == (0.1, 0.9, 5e-4)
    assert (cfg.dataset.batch_size, cfg.dataset.eval_batch_size) == (128, 256)
    assert cfg.dataset.num_streams == 2
    assert cfg.max_epochs == cfg.step.train_budget_epochs == 200
    assert isinstance(cfg.metrics_eval["test_acc"], BestAccuracy.Config)
    assert isinstance(cfg.metrics_eval["test_loss"], MeanCrossEntropy.Config)


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_every_experiment_states_its_hypothesis_sources_and_pending_results(
    factory: ExperimentFactory[CompressionTrainLoop.Config],
) -> None:
    doc = inspect.getdoc(factory) or ""
    assert "Hypothesis:" in doc
    assert "References:" in doc
    assert "80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c" in doc
    assert doc.rstrip().endswith("Results:\n  TBD.")
    assert factory.__name__ in (inspect.getdoc(experiments) or "")


def test_each_fork_names_its_parent_in_the_first_line() -> None:
    for child in ALL_EXPERIMENTS[1:]:
        assert (inspect.getdoc(child) or "").startswith("exp000 "), child.__name__


def test_exp000_matches_its_golden_config() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
