"""Tests for the ``cv-multitask-loss`` experiment chain.

``exp000`` is pinned by its delta from the vision recipe and by its golden;
each fork is checked as a delta, exactly which fields differ from ``exp000``,
so "one slot per idea" is a property of the code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import inspect

import pytest
import torch

from priml.baselines.mlsbench.multitask import experiments
from priml.baselines.mlsbench.multitask.experiments import (
    MultiTaskTrainLoop,
    exp000,
    exp001,
    exp002,
    exp003,
)
from priml.baselines.mlsbench.multitask.metrics import HeadAccuracy
from priml.baselines.mlsbench.multitask.model import TwoHeadResNet
from priml.baselines.mlsbench.pytorch_vision import experiments as vision
from priml.runtime import SingleProcess
from priml.testing.golden import assert_pprint_golden
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from pathlib import Path

    from priml.testing.experiments import ExperimentFactory

ALL_EXPERIMENTS: Final[list[ExperimentFactory[MultiTaskTrainLoop]]] = [
    exp000,
    exp001,
    exp002,
    exp003,
]


def changed_fields(parent: object, child: MultiTaskTrainLoop) -> set[str]:
    """Return the field paths ``udiff`` reports between two unfinalized configs."""
    assert isinstance(parent, MultiTaskTrainLoop | vision.VisionTrainLoop)
    table = parent.udiff(child, mode="fields", finalize=False)
    return {line.split("│")[0].strip() for line in table.splitlines()[1:]}


def shrink(config: MultiTaskTrainLoop, *, directory: Path) -> MultiTaskTrainLoop:
    """Narrow ``config`` to a size a CPU test can run, preserving its recipe."""
    assert isinstance(config.runtime, SingleProcess.Config)
    config.runtime.device = "cpu"
    config.step.parallelism = NoParallel.Config(device="cpu")
    assert isinstance(config.step.model, TwoHeadResNet.Config)
    config.step.model.trunk.blocks_per_stage = (1, 1, 1)
    config.step.model.trunk.channels_hidden = (2, 4, 5)
    config.step.train_budget_epochs = config.max_epochs = 1
    config.base_dir = None
    config.working_dir = directory
    config.dataset.data.working_dir = directory
    config.dataset.data.batch_size = config.dataset.data.eval_batch_size = 3
    config.dataset.data.translate_pad = 1
    config.checkpointer = None
    config.tracker = None
    return config


def write_dataset(directory: Path) -> None:
    """Write six 6x7 images per split in the prepared format."""
    for split in ("train", "test"):
        torch.save(
            {
                "media": (torch.arange(6 * 3 * 6 * 7) % 251)
                .reshape(6, 3, 6, 7)
                .to(torch.uint8),
                "label": torch.tensor([0, 17, 42, 99, 5, 63]),
            },
            directory / f"{split}.pt",
        )


def test_exp000_is_the_vision_recipe_with_a_second_head() -> None:
    assert changed_fields(vision.exp000(), exp000()) >= {"study_name"}
    cfg = exp000()
    assert isinstance(cfg.step.model, TwoHeadResNet.Config)
    assert cfg.step.model.trunk == vision.exp000().step.model
    assert cfg.step.model.channels_out == 20
    assert cfg.step.optimizer == vision.exp000().step.optimizer
    assert cfg.step.loss == vision.exp000().step.loss
    assert cfg.step.schedule == vision.exp000().step.schedule
    assert cfg.max_epochs == cfg.step.train_budget_epochs == 200
    accuracy = cfg.metrics_eval["test_acc_coarse"]
    assert isinstance(accuracy, HeadAccuracy.Config)
    assert (accuracy.logits_start, accuracy.logits_stop) == (100, 120)


@pytest.mark.parametrize(
    ("factory", "changed"),
    [
        (exp001, {"step.mtl_loss"}),
        (exp002, {"step.mtl_loss", "step.mtl_loss.temperature", "step.mtl_loss.eps"}),
        (exp003, {"step.mtl_loss", "step.mtl_loss.eps"}),
    ],
    ids=["exp001", "exp002", "exp003"],
)
def test_a_fork_changes_only_the_combination(
    factory: ExperimentFactory[MultiTaskTrainLoop],
    changed: set[str],
) -> None:
    assert changed_fields(exp000(), factory()) == {"experiment_name", *changed}


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_experiment_trains_at_minimum_size(
    factory: ExperimentFactory[MultiTaskTrainLoop],
    tmp_path: Path,
) -> None:
    write_dataset(tmp_path)
    loop = shrink(factory(), directory=tmp_path).make()
    loop.train()
    assert loop.step.global_step == 2
    assert factory().experiment_name == factory.__name__
    for name in ("test_acc", "test_acc_coarse"):
        accuracy = loop.metrics_eval[name]
        assert isinstance(accuracy, HeadAccuracy)
        assert accuracy.total == 6


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_every_experiment_states_its_hypothesis_sources_and_pending_results(
    factory: ExperimentFactory[MultiTaskTrainLoop],
) -> None:
    doc = inspect.getdoc(factory) or ""
    assert "Hypothesis:" in doc
    assert "80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c" in doc
    assert doc.rstrip().endswith("Results:\n  TBD.")
    assert factory.__name__ in (inspect.getdoc(experiments) or "")


def test_each_fork_names_its_parent_in_the_first_line() -> None:
    for child in (exp001, exp002, exp003):
        assert (inspect.getdoc(child) or "").startswith("exp000 + "), child.__name__


def test_exp000_matches_its_golden_config() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
