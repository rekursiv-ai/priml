"""Tests for the ``cv-sample-weighting`` experiment chain.

``exp000`` is pinned by its delta from the vision recipe and by its golden;
each fork is checked as a delta, exactly which fields differ from ``exp000``,
so "one slot per idea" is a property of the code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import inspect

import pytest
import torch

from priml.baselines.mlsbench.dl.vision import (
    experiments as vision_experiments,
    settings,
)
from priml.baselines.mlsbench.dl.vision.experiments_test import (
    changed_fields,
    shrink,
)
from priml.baselines.mlsbench.dl.weighting import experiments, reference
from priml.baselines.mlsbench.dl.weighting.data import LongTailCifar100Data
from priml.baselines.mlsbench.dl.weighting.experiments import (
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
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from pathlib import Path

    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop
    from priml.testing.experiments import ExperimentFactory

ALL_EXPERIMENTS: Final[list[ExperimentFactory[VisionTrainLoop]]] = [
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

INVERSE: Final = {"step.loss.class_weight.func"}
EFFECTIVE: Final = {"step.loss.class_weight.func", "step.loss.class_weight.beta"}
GAP: Final = {"step.loss.class_weight.func", "step.loss.class_weight.spread"}

FORKS: Final = {
    exp001: (exp000, INVERSE),
    exp002: (exp000, EFFECTIVE),
    exp003: (exp000, GAP),
    exp005: (exp004, INVERSE),
    exp006: (exp004, EFFECTIVE),
    exp007: (exp004, GAP),
    exp009: (exp008, INVERSE),
    exp010: (exp008, EFFECTIVE),
    exp011: (exp008, GAP),
}
"""Each baseline fork, its setting's uniform parent, and the fields it sets."""


def test_exp000_is_the_vision_recipe_at_resnet32_on_the_long_tail() -> None:
    assert changed_fields(vision_experiments.exp000(), exp000()) == {
        "study_name",
        *(f"step.model.blocks_per_stage[{stage}]" for stage in range(3)),
        "step.loss",
        "step.loss.func",
        "step.loss.class_weight",
        "step.loss.class_weight.func",
        "dataset",
        "dataset.imbalance_ratio",
        "dataset.num_classes",
        "dataset.subset_seed",
    }
    cfg = exp000().copy_tree().finalize()
    assert cfg.dataset.batch_size == 128
    assert cfg.max_epochs == cfg.step.train_budget_epochs == 200


@pytest.mark.parametrize(
    ("factory", "parent", "changed"),
    [(child, parent, changed) for child, (parent, changed) in FORKS.items()],
    ids=[f.__name__ for f in FORKS],
)
def test_a_fork_changes_only_the_class_weight(
    factory: ExperimentFactory[VisionTrainLoop],
    parent: ExperimentFactory[VisionTrainLoop],
    changed: set[str],
) -> None:
    assert changed_fields(parent(), factory()) == {"experiment_name", *changed}


def test_exp004_is_exp000_on_cifar10() -> None:
    assert changed_fields(exp000(), exp004()) == {
        "experiment_name",
        "step.model.channels_out",
        "dataset.num_classes",
        "dataset.working_dir",
        *(f"dataset.{name}[{index}]" for name in ("mean", "std") for index in range(3)),
    }
    assert (
        exp004().dataset.mean
        == settings.resnet20_cifar10(vision_experiments.exp000()).dataset.mean
    )


def test_exp008_is_the_vgg16bn_setting_at_ratio_50() -> None:
    expected = settings.vgg16bn_cifar100(exp000())
    expected.experiment_name = "exp008"
    assert changed_fields(expected, exp008()) == {"dataset.imbalance_ratio"}


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_experiment_trains_at_minimum_size(
    factory: ExperimentFactory[VisionTrainLoop],
    tmp_path: Path,
) -> None:
    media, label = reference.long_tail_split()
    for split in ("train", "test"):
        torch.save({"media": media, "label": label}, tmp_path / f"{split}.pt")
    cfg = shrink(factory(), directory=tmp_path)
    assert isinstance(cfg.dataset, LongTailCifar100Data.Config)
    cfg.dataset.imbalance_ratio = reference.SPLIT_RATIO
    cfg.dataset.batch_size = cfg.dataset.eval_batch_size = 6
    loop = cfg.make()
    loop.train()
    # 4.0**-33 over 100 classes keeps 6, 3, 2 and 1 of the four held: 12 images.
    assert loop.step.global_step == 2
    assert factory().experiment_name == factory.__name__


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_every_experiment_states_its_hypothesis_sources_and_pending_results(
    factory: ExperimentFactory[VisionTrainLoop],
) -> None:
    doc = inspect.getdoc(factory) or ""
    assert "Hypothesis:" in doc
    assert "b873944969c53b01e4ee6e843c77767b8c702485" in doc
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


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
