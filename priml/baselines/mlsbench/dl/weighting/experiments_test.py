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

from priml.baselines.mlsbench.dl.vision import experiments as vision
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
]


def test_exp000_is_the_vision_recipe_at_resnet32_on_the_long_tail() -> None:
    assert changed_fields(vision.exp000(), exp000()) == {
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
    ("factory", "changed"),
    [
        (exp001, {"step.loss.class_weight.func"}),
        (exp002, {"step.loss.class_weight.func", "step.loss.class_weight.beta"}),
        (exp003, {"step.loss.class_weight.func", "step.loss.class_weight.spread"}),
    ],
    ids=["exp001", "exp002", "exp003"],
)
def test_a_fork_changes_only_the_class_weight(
    factory: ExperimentFactory[VisionTrainLoop],
    changed: set[str],
) -> None:
    assert changed_fields(exp000(), factory()) == {"experiment_name", *changed}


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
