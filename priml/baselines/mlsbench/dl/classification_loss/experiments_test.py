"""Tests for the ``cv-classification-loss`` experiment chain.

``exp000`` is pinned by value and by its golden. Each fork is checked as a
DELTA, exactly which fields differ from its parent, which keeps "one change per
experiment" a property of the code rather than a claim in a docstring.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Final, cast

import inspect

from configgle import InlineConfig
from treekle.absent import ABSENT

import pytest

from priml.baselines.mlsbench.dl.classification_loss import experiments
from priml.baselines.mlsbench.dl.classification_loss.experiments import (
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
    experiments as vision_experiments,
    settings,
)
from priml.baselines.mlsbench.dl.vision.experiments_test import (
    changed_fields,
    shrink,
    write_dataset,
)
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from _typeshed import DataclassInstance

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

FORKS: Final[dict[str, tuple[ExperimentFactory[VisionTrainLoop], ...]]] = {
    "exp000": (exp001, exp002, exp003, exp004, exp008),
    "exp004": (exp005, exp006, exp007),
    "exp008": (exp009, exp010, exp011),
}
"""Each parent's direct forks, so every first line names its parent."""

LOSS_DELTAS: Final = [
    {"step.loss.func", "step.loss.smoothing"},
    {"step.loss.func", "step.loss.gamma"},
    {"step.loss.func", "step.loss.epsilon"},
]
"""The fields the label-smoothing, focal and PolyLoss forks change, in that order."""


def test_exp000_is_the_vision_recipe() -> None:
    assert changed_fields(vision_experiments.exp000(), exp000()) == {"study_name"}


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_experiment_trains_at_minimum_size(
    factory: ExperimentFactory[VisionTrainLoop],
    tmp_path: Path,
) -> None:
    # Ten labels fit every setting's head: Fashion-MNIST's has ten classes.
    write_dataset(tmp_path, classes=10)
    loop = shrink(factory(), directory=tmp_path).make()
    loop.train()
    assert loop.step.global_step == 2
    # The launcher derives the output directory from this field; a mismatch
    # silently writes one run over another.
    assert factory().experiment_name == factory.__name__


@pytest.mark.parametrize(
    ("parent", "factory", "changed"),
    [
        (parent, factory, changed)
        for parent, forks in (
            (exp000, (exp001, exp002, exp003)),
            (exp004, (exp005, exp006, exp007)),
            (exp008, (exp009, exp010, exp011)),
        )
        for factory, changed in zip(forks, LOSS_DELTAS, strict=True)
    ],
    ids=[f"exp{n:03d}" for n in (1, 2, 3, 5, 6, 7, 9, 10, 11)],
)
def test_a_fork_changes_only_the_loss(
    parent: ExperimentFactory[VisionTrainLoop],
    factory: ExperimentFactory[VisionTrainLoop],
    changed: set[str],
) -> None:
    assert changed_fields(parent(), factory()) == {"experiment_name", *changed}
    assert _deltas(parent(), factory()) == {"experiment_name", "step.loss"}


@pytest.mark.parametrize(
    ("factory", "switch"),
    [
        (exp004, settings.vgg16bn_cifar100),
        (exp008, settings.mobilenetv2_fmnist),
    ],
    ids=["exp004", "exp008"],
)
def test_a_setting_fork_changes_only_the_network_and_the_dataset(
    factory: ExperimentFactory[VisionTrainLoop],
    switch: Callable[[VisionTrainLoop], VisionTrainLoop],
) -> None:
    changed = changed_fields(exp000(), factory())
    assert changed == changed_fields(exp000(), switch(exp000())) | {"experiment_name"}
    assert all(
        path == "experiment_name" or path.startswith(("step.model", "dataset."))
        for path in changed
    )
    assert factory().step.loss == exp000().step.loss


def test_forks_do_not_mutate_their_parent() -> None:
    for parent, forks in ((exp000, (exp003, exp004)), (exp004, (exp007,))):
        baseline = parent()
        before = _flatten(baseline)
        for fork in forks:
            _ = fork()
        assert _flatten(baseline) == before


def test_each_fork_names_its_parent_in_the_first_line() -> None:
    for parent, children in FORKS.items():
        for child in children:
            assert parent in (inspect.getdoc(child) or "").splitlines()[0], (
                child.__name__
            )


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
    assert "References:" in doc
    assert "80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c" in doc
    assert doc.rstrip().endswith("Results:\n  TBD.")
    assert factory.__name__ in (inspect.getdoc(experiments) or "")


def test_exp000_matches_its_golden_config() -> None:
    """Pin the WHOLE finalized ``exp000``: the control every fork is measured against."""
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


# A field holding a nested Config is descended into, so a change buried in
# ``step.model`` is reported at the leaf that actually moved; an injected callable is
# compared by its repr, which carries the function and its bound arguments.
def _deltas(parent: VisionTrainLoop, child: VisionTrainLoop) -> set[str]:
    """Return the dotted names of the fields that differ between two configs."""
    flat_parent, flat_child = _flatten(parent), _flatten(child)
    return {
        name
        for name in flat_parent.keys() | flat_child.keys()
        if flat_parent.get(name, ABSENT) != flat_child.get(name, ABSENT)
    }


def _flatten(config: DataclassInstance, prefix: str = "") -> dict[str, object]:
    """Return a dotted-name to value map, descending into nested Configs."""
    flat: dict[str, object] = {}
    for field in fields(config):
        value = cast(object, getattr(config, field.name))
        dotted = f"{prefix}{field.name}"
        if isinstance(value, InlineConfig):
            flat[dotted] = repr(cast("InlineConfig[object]", value))
        elif isinstance(value, dict):
            flat[dotted] = {
                k: repr(v) for k, v in cast("dict[str, object]", value).items()
            }
        elif is_dataclass(value):
            flat[dotted] = type(value)
            flat.update(_flatten(cast("DataclassInstance", value), prefix=f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
