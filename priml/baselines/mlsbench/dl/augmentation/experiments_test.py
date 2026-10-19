"""Tests for the ``cv-data-augmentation`` experiment chain.

``exp000`` is pinned by value and by its golden; each fork is checked as a
delta, exactly which fields differ from ``exp000``, so "one slot per idea" is a
property of the code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import inspect

import pytest
import torch

from priml.baselines.mlsbench.dl.augmentation import experiments
from priml.baselines.mlsbench.dl.augmentation.experiments import (
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
from priml.baselines.mlsbench.dl.vision import (
    experiments as vision,
    settings,
)
from priml.baselines.mlsbench.dl.vision.experiments_test import (
    changed_fields,
    shrink,
    write_dataset,
)
from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from collections.abc import Callable
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

AUGMENT_DELTAS: Final = {
    "cutout": {
        "dataset.augment.func",
        "dataset.augment.holes",
        "dataset.augment.length",
    },
    "randaugment": {
        "dataset.augment.func",
        "dataset.augment.num_ops",
        "dataset.augment.magnitude",
        "dataset.augment.num_magnitude_bins",
    },
    "trivialaugment": {"dataset.augment.func", "dataset.augment.num_magnitude_bins"},
}
"""What each baseline's fork changes on top of its setting's experiment."""


def test_exp000_is_the_vision_recipe_under_this_tasks_name() -> None:
    assert changed_fields(vision.exp000(), exp000()) == {"study_name"}


@pytest.mark.parametrize(
    ("parent", "factory", "changed"),
    [
        (exp000, exp001, AUGMENT_DELTAS["cutout"]),
        (exp000, exp002, AUGMENT_DELTAS["randaugment"]),
        (exp000, exp003, AUGMENT_DELTAS["trivialaugment"]),
        (exp004, exp005, AUGMENT_DELTAS["cutout"]),
        (exp004, exp006, AUGMENT_DELTAS["randaugment"]),
        (exp004, exp007, AUGMENT_DELTAS["trivialaugment"]),
        (exp008, exp009, AUGMENT_DELTAS["cutout"]),
        (exp008, exp010, AUGMENT_DELTAS["randaugment"]),
        (exp008, exp011, AUGMENT_DELTAS["trivialaugment"]),
    ],
    ids=[f"exp{n:03d}" for n in (1, 2, 3, 5, 6, 7, 9, 10, 11)],
)
def test_a_fork_changes_only_the_augmentation(
    parent: ExperimentFactory[VisionTrainLoop],
    factory: ExperimentFactory[VisionTrainLoop],
    changed: set[str],
) -> None:
    assert changed_fields(parent(), factory()) == {"experiment_name", *changed}


@pytest.mark.parametrize(
    ("factory", "switch"),
    [
        (exp004, settings.resnet20_cifar10),
        (exp008, settings.mobilenetv2_fmnist),
    ],
    ids=["exp004", "exp008"],
)
def test_a_setting_fork_is_exp000_switched_and_renamed(
    factory: ExperimentFactory[VisionTrainLoop],
    switch: Callable[[VisionTrainLoop], VisionTrainLoop],
) -> None:
    switched = switch(exp000())
    assert changed_fields(switched, factory()) == {"experiment_name"}
    changed = changed_fields(exp000(), factory())
    assert all(
        path == "experiment_name" or path.startswith(("step.model", "dataset."))
        for path in changed
    )
    assert not any(path.startswith("dataset.augment") for path in changed)


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_experiment_trains_at_minimum_size(
    factory: ExperimentFactory[VisionTrainLoop],
    tmp_path: Path,
) -> None:
    config = factory()
    model = config.step.model
    assert isinstance(model, ResNet.Config | MobileNetV2.Config)
    write_dataset(tmp_path, classes=model.channels_out)
    if config.dataset.working_dir == "/datasets/fmnist":
        # The prepared Fashion-MNIST cache holds one channel, so the slot does too.
        for split in ("train", "test"):
            path = tmp_path / f"{split}.pt"
            data = cast("dict[str, torch.Tensor]", torch.load(path, weights_only=True))
            torch.save({"media": data["media"][:, :1], "label": data["label"]}, path)
    loop = shrink(config, directory=tmp_path).make()
    loop.train()
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


@pytest.mark.parametrize(
    ("parent", "children"),
    [
        ("exp000", (exp001, exp002, exp003)),
        ("exp004", (exp005, exp006, exp007)),
        ("exp008", (exp009, exp010, exp011)),
    ],
)
def test_each_fork_names_its_parent_in_the_first_line(
    parent: str,
    children: tuple[ExperimentFactory[VisionTrainLoop], ...],
) -> None:
    for child in children:
        assert (inspect.getdoc(child) or "").startswith(f"{parent} + "), child.__name__
    for setting in (exp004, exp008):
        assert (inspect.getdoc(setting) or "").startswith("exp000 at the ")


def test_exp000_matches_its_golden_config() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
