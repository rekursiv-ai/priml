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

from priml.baselines.mlsbench.dl.multitask import experiments
from priml.baselines.mlsbench.dl.multitask.experiments import (
    MultiTaskTrainLoop,
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp006,
    exp007,
    exp008,
    exp009,
    exp010,
    exp011,
)
from priml.baselines.mlsbench.dl.multitask.metrics import HeadAccuracy
from priml.baselines.mlsbench.dl.multitask.model import TwoHeadResNet
from priml.baselines.mlsbench.dl.vision import (
    experiments as vision,
    settings,
)
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN
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
    exp004,
    exp005,
    exp006,
    exp007,
    exp008,
    exp009,
    exp010,
    exp011,
]

UNCERTAINTY: Final = {"step.mtl_loss"}
DWA: Final = {"step.mtl_loss", "step.mtl_loss.temperature", "step.mtl_loss.eps"}
PCGRAD: Final = {"step.mtl_loss", "step.mtl_loss.eps"}

FORKS: Final = {
    exp001: (exp000, UNCERTAINTY),
    exp002: (exp000, DWA),
    exp003: (exp000, PCGRAD),
    exp005: (exp004, UNCERTAINTY),
    exp006: (exp004, DWA),
    exp007: (exp004, PCGRAD),
    exp009: (exp008, UNCERTAINTY),
    exp010: (exp008, DWA),
    exp011: (exp008, PCGRAD),
}
"""Each baseline fork, its setting's summing parent, and the fields it sets."""


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
    trunk = config.step.model.trunk
    if isinstance(trunk, VGG16BN.Config):
        trunk.channels_hidden = ((4,), (5,))
        trunk.channels_hidden_head = 6
    else:
        assert isinstance(trunk, ResNet.Config)
        trunk.blocks_per_stage = (1, 1, 1)
        trunk.channels_hidden = (2, 4, 5)
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
    """Write six 5x7 images per split in the prepared format."""
    for split in ("train", "test"):
        torch.save(
            {
                "media": (torch.arange(6 * 3 * 5 * 7) % 251)
                .reshape(6, 3, 5, 7)
                .to(torch.uint8),
                "label": torch.tensor([0, 17, 42, 99, 5, 63]),
            },
            directory / f"{split}.pt",
        )


MULTITASK_SLOTS: Final = (
    "dataset",
    "metrics_eval",
    "step",
)
"""Where the multitask recipe differs from the vision one: data, heads, metrics.

``step`` covers the two-head model and the combination; the trunk is pinned
below to equal the vision network, so the delta cannot hide a trunk change."""


def test_exp000_is_the_vision_recipe_with_a_second_head() -> None:
    changed = changed_fields(vision.exp000(), exp000())
    assert {p for p in changed if not p.startswith(MULTITASK_SLOTS)} == {
        "(class)",
        "study_name",
    }
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
    ("factory", "parent", "changed"),
    [(child, parent, changed) for child, (parent, changed) in FORKS.items()],
    ids=[f.__name__ for f in FORKS],
)
def test_a_fork_changes_only_the_combination(
    factory: ExperimentFactory[MultiTaskTrainLoop],
    parent: ExperimentFactory[MultiTaskTrainLoop],
    changed: set[str],
) -> None:
    assert changed_fields(parent(), factory()) == {"experiment_name", *changed}


def test_exp004_deepens_nothing_but_the_trunk_to_resnet20() -> None:
    assert changed_fields(exp000(), exp004()) == {
        "experiment_name",
        *(f"step.model.trunk.blocks_per_stage[{index}]" for index in range(3)),
    }


def test_exp008_puts_the_vgg16bn_setting_network_in_the_trunk() -> None:
    changed = changed_fields(exp000(), exp008())
    assert {p for p in changed if not p.startswith("step.model.trunk")} == {
        "experiment_name",
    }
    model = exp008().step.model
    assert isinstance(model, TwoHeadResNet.Config)
    assert model.trunk == settings.vgg16bn_cifar100(vision.exp000()).step.model
    assert exp008().dataset == exp000().dataset


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
    for child, (parent, _) in FORKS.items():
        assert (inspect.getdoc(child) or "").startswith(f"{parent.__name__} + "), (
            child.__name__
        )
    for setting in (exp004, exp008):
        assert (inspect.getdoc(setting) or "").startswith("exp000 at the "), (
            setting.__name__
        )


def test_exp000_matches_its_golden_config() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


def test_a_loop_that_would_leave_the_references_draws_is_refused() -> None:
    config = exp000()
    config.eval_warmup_batches = 1
    with pytest.raises(ValueError, match="eval_warmup_batches"):
        config.copy_tree().finalize()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
