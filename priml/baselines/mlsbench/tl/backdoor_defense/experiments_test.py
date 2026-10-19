"""Tests for the ``security-backdoor-defense`` experiment chain.

``exp000`` is pinned by value and by its golden; each fork is checked as a
delta -- exactly which fields differ from ``exp000`` -- so a baseline changes
only the defense and a setting only the network, the data and the trigger.
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
    write_dataset,
)
from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN
from priml.baselines.mlsbench.tl.backdoor_defense import experiments
from priml.baselines.mlsbench.tl.backdoor_defense.defenses import MaxConfidence
from priml.baselines.mlsbench.tl.backdoor_defense.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp006,
)
from priml.baselines.mlsbench.tl.backdoor_defense.triggers import badnets, blend
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from pathlib import Path

    from priml.baselines.mlsbench.tl.backdoor_defense.train_loop import (
        BackdoorTrainLoop,
    )
    from priml.testing.experiments import ExperimentFactory

ALL_EXPERIMENTS: Final[list[ExperimentFactory[BackdoorTrainLoop.Config]]] = [
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp006,
]

POISONING: Final = ("trigger", "poison_fraction", "target_class", "poison_seed")
"""The fields ``PoisonedData.Config`` adds to the vision dataset."""


def test_exp000_is_the_vision_recipe_at_resnet20_cifar10_for_100_epochs() -> None:
    parent = settings.resnet20_cifar10(vision_experiments.exp000())
    changed = {
        field
        for field in changed_fields(parent, exp000())
        if not field.startswith(("checkpointer.", "eval_epochs.", "dataset."))
    }
    # The step, the network and the dataset's statistics are the vision
    # setting's; the loop's class, horizon and the task's own slots are new.
    assert changed == {
        "(class)",
        "study_name",
        "max_epochs",
        "step.train_budget_epochs",
        "dataset",
        "checkpointer",
        "defense",
        "eval_epochs",
        "early_stopping",
        "filter_multiplier",
    }
    poisoning = {
        field
        for field in changed_fields(parent, exp000())
        if field.startswith("dataset.")
    }
    assert poisoning == {f"dataset.{name}" for name in POISONING}


def test_exp000_pins_the_driver_recipe() -> None:
    cfg = exp000()
    assert isinstance(cfg.step.model, ResNet.Config)
    assert cfg.step.model.blocks_per_stage == (3, 3, 3)
    assert cfg.step.model.channels_out == 10
    assert cfg.max_epochs == cfg.step.train_budget_epochs == 100
    assert cfg.dataset.trigger is badnets
    assert cfg.dataset.poison_fraction == 0.05
    assert (cfg.dataset.target_class, cfg.dataset.poison_seed) == (0, 42)
    assert cfg.filter_multiplier == 1.5
    assert isinstance(cfg.defense, MaxConfidence.Config)
    assert cfg.checkpointer is None
    evaluate = cfg.eval_epochs.make()
    assert [e for e in range(1, 101) if evaluate(e, total_epochs=100)] == [
        1,
        *range(10, 101, 10),
    ]


@pytest.mark.parametrize(
    ("factory", "fields"),
    [
        (exp001, {"eps"}),
        (exp002, set[str]()),
        (exp003, {"eps", "max_iter", "membership", "min_class_size"}),
        (exp004, {"eps"}),
    ],
    ids=["exp001", "exp002", "exp003", "exp004"],
)
def test_a_baseline_changes_only_the_defense(
    factory: ExperimentFactory[BackdoorTrainLoop.Config],
    fields: set[str],
) -> None:
    # A defense's own fields read as changes: the pristine one has none.
    assert changed_fields(exp000(), factory()) == {
        "experiment_name",
        "defense",
        *(f"defense.{name}" for name in fields),
    }


@pytest.mark.parametrize(
    ("factory", "model", "trigger", "fraction"),
    [
        (exp005, VGG16BN.Config, blend, 0.01),
        (exp006, MobileNetV2.Config, badnets, 0.08),
    ],
    ids=["exp005", "exp006"],
)
def test_a_setting_changes_only_the_network_the_data_and_the_trigger(
    factory: ExperimentFactory[BackdoorTrainLoop.Config],
    model: type,
    trigger: object,
    fraction: float,
) -> None:
    cfg = factory()
    assert isinstance(cfg.step.model, model)
    assert cfg.dataset.trigger is trigger
    assert cfg.dataset.poison_fraction == fraction
    allowed = {"experiment_name", "dataset.working_dir", "dataset.trigger"}
    allowed |= {"dataset.poison_fraction", "dataset.mean", "dataset.std"}
    changed = {
        field.split("[")[0]
        for field in changed_fields(exp000(), cfg)
        if not field.startswith("step.model")
    }
    assert changed <= allowed
    assert "dataset.poison_fraction" in changed


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_every_experiment_states_its_hypothesis_sources_and_pending_results(
    factory: ExperimentFactory[BackdoorTrainLoop.Config],
) -> None:
    doc = inspect.getdoc(factory) or ""
    assert doc.startswith("exp000 ") or factory is exp000
    assert "Hypothesis:" in doc
    assert "References:" in doc
    assert "80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c" in doc
    assert doc.rstrip().endswith("Results:\n  TBD.")
    assert factory.__name__ in (inspect.getdoc(experiments) or "")
    assert factory().experiment_name == factory.__name__


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_every_experiment_filters_and_retrains_at_minimum_size(
    factory: ExperimentFactory[BackdoorTrainLoop.Config],
    tmp_path: Path,
) -> None:
    write_dataset(tmp_path, classes=10)
    cfg = shrink(factory(), directory=tmp_path)
    cfg.checkpointer = None
    cfg.dataset.poison_fraction = 0.25
    loop = cfg.make()
    loop.train()
    assert loop.filtered is not None
    assert loop.report is not None
    # int(8 * 0.25 * 1.5) = 3 of the 8 training images are removed.
    assert len(loop.filtered.removed) == 3
    assert len(loop.filtered.kept) == 5
    assert 0 <= loop.report.asr <= 1
    assert 0 <= loop.report.clean_acc <= 1


def test_a_checkpointer_is_refused(tmp_path: Path) -> None:
    write_dataset(tmp_path, classes=10)
    cfg = shrink(exp000(), directory=tmp_path)
    cfg.checkpointer = vision_experiments.exp000().checkpointer
    with pytest.raises(ValueError, match="checkpointer=None"):
        cfg.make()


def test_construction_reads_no_files(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("experiment construction must not load tensors")

    monkeypatch.setattr(torch, "load", boom)
    for factory in ALL_EXPERIMENTS:
        _ = factory().copy_tree().finalize()


def test_exp000_matches_its_golden_config() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
